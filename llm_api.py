"""
Minimal client for an OpenAI-compatible chat-completions API (for example, ChatAvalAI).

Configuration is read from environment variables so that no secret is stored in code:
    CHATAVALAI_BASE_URL  e.g. https://<provider-host>/v1   (the path /chat/completions is appended)
    CHATAVALAI_API_KEY   your API token
    CHATAVALAI_MODEL     model name exposed by the provider (optional; the provider default is used if empty)

Only the Python standard library is used.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional


class LLMError(RuntimeError):
    """Raised when the API is not configured or returns an error that cannot be retried."""


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str = ""
    timeout: float = 60.0
    max_retries: int = 3
    backoff_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Build a configuration from environment variables.

        Raises:
            LLMError: If the URL or the API key is missing.
        """
        base_url = os.environ.get("https://api.avalai.ir/v1", "").strip()
        api_key = os.environ.get("aa-jntrmr21hYGICbecUOFWM3hDFk8NqlqgtL96dPSqDWGxTXCS", "").strip()
        missing = [name for name, value in (("https://api.avalai.ir/v1", base_url),
                                             ("aa-jntrmr21hYGICbecUOFWM3hDFk8NqlqgtL96dPSqDWGxTXCS", api_key)) if not value]
        if missing:
            raise LLMError("missing environment variable(s): " + ", ".join(missing))
        return cls(base_url=base_url, api_key=api_key, model=os.environ.get("CHATAVALAI_MODEL", "").strip())


class LLMClient:
    """Sends chat requests and returns the assistant's text."""

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, config: LLMConfig, opener: Optional[urllib.request.OpenerDirector] = None):
        """
        Args:
            config (LLMConfig): Endpoint and credentials.
            opener (OpenerDirector, optional): Custom urllib opener (used by the tests).
        """
        self.config = config
        self.opener = opener or urllib.request.build_opener()

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> str:
        """
        Send one chat request and return the assistant message text.

        Args:
            messages (List[Dict[str, str]]): Messages with 'role' and 'content'.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens in the reply.

        Returns:
            str: Assistant reply text.

        Raises:
            LLMError: On authentication errors, malformed responses, or exhausted retries.
        """
        payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if self.config.model:
            payload["model"] = self.config.model
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            try:
                with self.opener.open(request, timeout=self.config.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return self._extract_text(data)
            except urllib.error.HTTPError as e:
                if e.code not in self.RETRYABLE_STATUS:
                    raise LLMError(f"API returned HTTP {e.code}: {e.reason}") from e
                last_error = e
            except urllib.error.URLError as e:
                last_error = e
            if attempt < self.config.max_retries:
                time.sleep(self.config.backoff_seconds * (2 ** attempt))
        raise LLMError(f"request failed after {self.config.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response format: {str(data)[:200]}") from e


# ===== UNIT TESTS (local mock server, no external network) =====

def _start_mock_server(fail_first: int = 0, status_for_failure: int = 500, expected_key: str = "test-key"):
    """Start a local OpenAI-style server. Returns (server, received) where received collects requests."""
    import http.server
    import threading

    received: List[dict] = []
    state = {"failures_left": fail_first}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            auth = self.headers.get("Authorization", "")
            received.append({"path": self.path, "auth": auth, "payload": payload})
            if auth != f"Bearer {expected_key}":
                self._send(401, {"error": "bad key"})
            elif state["failures_left"] > 0:
                state["failures_left"] -= 1
                self._send(status_for_failure, {"error": "try again"})
            else:
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": "hello from mock"}}]})

        def _send(self, code, obj):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def _client_for(server, key: str = "test-key", retries: int = 3) -> LLMClient:
    port = server.server_address[1]
    config = LLMConfig(base_url=f"http://127.0.0.1:{port}/v1", api_key=key, model="demo-model",
                       timeout=5.0, max_retries=retries, backoff_seconds=0.01)
    # Bypass system proxies so that the request reaches the local server directly.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return LLMClient(config, opener=opener)


def test_chat_success_and_request_shape():
    server, received = _start_mock_server()
    try:
        reply = _client_for(server).chat([{"role": "user", "content": "سلام"}], temperature=0.2, max_tokens=50)
        assert reply == "hello from mock"
        req = received[-1]
        assert req["path"] == "/v1/chat/completions", "Endpoint must append /chat/completions."
        assert req["auth"] == "Bearer test-key"
        assert req["payload"]["model"] == "demo-model"
        assert req["payload"]["messages"][0]["content"] == "سلام", "Unicode must survive the round trip."
    finally:
        server.shutdown()


def test_retries_on_server_error_then_succeeds():
    server, received = _start_mock_server(fail_first=2, status_for_failure=503)
    try:
        assert _client_for(server).chat([{"role": "user", "content": "x"}]) == "hello from mock"
        assert len(received) == 3, "Two 503 responses should be retried before the success."
    finally:
        server.shutdown()


def test_bad_key_is_not_retried():
    server, received = _start_mock_server()
    try:
        try:
            _client_for(server, key="wrong").chat([{"role": "user", "content": "x"}])
        except LLMError as e:
            assert "401" in str(e)
        else:
            raise AssertionError("A 401 response must raise LLMError.")
        assert len(received) == 1, "Authentication errors must not be retried."
    finally:
        server.shutdown()


def test_from_env_requires_both_variables(monkeypatch=None):
    saved = {k: os.environ.pop(k, None) for k in ("https://api.avalai.ir/v1", "aa-jntrmr21hYGICbecUOFWM3hDFk8NqlqgtL96dPSqDWGxTXCS")}
    try:
        try:
            LLMConfig.from_env()
        except LLMError as e:
            assert "https://api.avalai.ir/v1" in str(e) and "aa-jntrmr21hYGICbecUOFWM3hDFk8NqlqgtL96dPSqDWGxTXCS" in str(e)
        else:
            raise AssertionError("Missing variables must raise LLMError.")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def run_all_tests() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all_tests()