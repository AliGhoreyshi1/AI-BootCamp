"""
Connects the project pieces:

    LLM API (llm_api)  --documents-->  data_pipeline  --tokens-->  mini_gpt_torch.MiniGPT
                                                                         |
                              side-by-side comparison with the API model  <-+

The API is optional. Every function that needs it accepts a client, and the
comparison simply reports that no API is configured when none is available.
"""

import random
import time
from typing import Iterable, List, Optional, Sequence

import torch

from data_pipeline import build_training_data, generate_sample_corpus
from llm_api import LLMClient, LLMConfig, LLMError
from mini_gpt_torch import MiniGPT, cross_entropy_loss, generate

IGNORE_INDEX = -100
GREEDY_TEMPERATURE = 1e-6


# ---------------------------------------------------------------------------
# 1) Corpus from the API
# ---------------------------------------------------------------------------

def generate_corpus_with_llm(client, topics: Sequence[str], per_topic: int = 3,
                             language: str = "Persian") -> List[str]:
    """
    Ask the API for short texts on each topic. The texts then go through the data pipeline
    like any other raw documents (cleaning, language filter, quality filter, dedup, PII).

    Args:
        client: Any object with chat(messages, temperature, max_tokens) -> str.
        topics (Sequence[str]): Topics to write about.
        per_topic (int): Number of documents requested per topic.
        language (str): Language requested from the model.

    Returns:
        List[str]: Raw documents (not yet cleaned).
    """
    docs: List[str] = []
    for topic in topics:
        for i in range(per_topic):
            messages = [
                {"role": "system", "content": f"You write short, factual paragraphs in {language}."},
                {"role": "user", "content": f"Write one paragraph (4 to 6 sentences) about: {topic}. Variant {i + 1}."},
            ]
            docs.append(client.chat(messages, temperature=0.9, max_tokens=300))
    return docs


# ---------------------------------------------------------------------------
# 2) Train MiniGPT on pipeline output
# ---------------------------------------------------------------------------

def masked_loss(model: MiniGPT, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over the supervised positions only (labels != IGNORE_INDEX).

    The course loss has no ignore index, so the supervised logits are selected first and
    passed as a single batch. Gradients still flow through the selection.

    Args:
        model (MiniGPT): Model.
        inputs (Tensor): (batch, seq) token ids.
        labels (Tensor): (batch, seq) next-token labels with IGNORE_INDEX on unsupervised positions.

    Returns:
        Tensor: Scalar loss.
    """
    logits = model(inputs)
    keep = labels != IGNORE_INDEX
    if not bool(keep.any()):
        raise ValueError("no supervised positions in this batch")
    return cross_entropy_loss(logits[keep].unsqueeze(0), labels[keep].unsqueeze(0))


def train_on_pipeline(raw_docs: Iterable[str], block_size: int = 64, merges: int = 100,
                      steps: int = 600, batch_size: int = 8, lr: float = 3e-3, seed: int = 0,
                      embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2,
                      log_every: int = 100):
    """
    Build the dataset with data_pipeline and train the course MiniGPT on it.

    Args:
        raw_docs (Iterable[str]): Raw documents (from the API, a file, or the sample corpus).
        block_size (int): Packed block length, also the model's maximum sequence length.
        merges (int): BPE merges to learn.
        steps (int): Optimizer steps.
        batch_size (int): Blocks per step.
        lr (float): AdamW learning rate.
        seed (int): Seed for torch and for batch order.
        embed_dim, num_heads, num_layers (int): Model size.
        log_every (int): Print frequency in steps.

    Returns:
        Tuple[tokenizer, MiniGPT, List[float], List[str]]: tokenizer, trained model, loss history,
        final documents that were used for training.
    """
    torch.manual_seed(seed)
    tokenizer, packed, docs = build_training_data(raw_docs, num_merges=merges, seq_length=block_size)
    inputs = torch.tensor(packed["input_ids"], dtype=torch.long)
    labels = torch.tensor(packed["labels"], dtype=torch.long)

    model = MiniGPT(vocab_size=tokenizer.vocab_size(), embed_dim=embed_dim, num_heads=num_heads,
                    num_layers=num_layers, max_seq_len=block_size, ff_dim=4 * embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    losses: List[float] = []
    model.train()
    for step in range(1, steps + 1):
        pick = rng.sample(range(inputs.size(0)), k=min(batch_size, inputs.size(0)))
        loss = masked_loss(model, inputs[pick], labels[pick])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            print(f"step {step:5d} | loss {losses[-1]:.4f}")
    model.eval()
    return tokenizer, model, losses, docs


# ---------------------------------------------------------------------------
# 3) Comparison with the API model
# ---------------------------------------------------------------------------

def compare_with_llm(tokenizer, model: MiniGPT, prompts: Sequence[str],
                     client: Optional[LLMClient] = None, max_new_tokens: int = 40) -> List[dict]:
    """
    For each prompt, produce a continuation from MiniGPT and (if configured) from the API model.

    Args:
        tokenizer: Trained tokenizer (encode/decode).
        model (MiniGPT): Trained model.
        prompts (Sequence[str]): Prompts.
        client (LLMClient, optional): API client. If None, only MiniGPT is run.
        max_new_tokens (int): Continuation length for MiniGPT (in tokens).

    Returns:
        List[dict]: One record per prompt with keys 'prompt', 'minigpt', 'api' (None if unavailable).
    """
    records = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        # The course generate() divides by temperature, so 'greedy' is a near-zero temperature.
        ids = generate(model, prompt_ids, max_new_tokens=max_new_tokens, temperature=GREEDY_TEMPERATURE)
        mini_text = tokenizer.decode(ids[len(prompt_ids):])

        api_text = None
        if client is not None:
            try:
                api_text = client.chat(
                    [{"role": "user", "content": f"Continue this text naturally: {prompt}"}],
                    temperature=0.0, max_tokens=max_new_tokens * 2)
            except LLMError as e:
                api_text = f"[API error: {e}]"
        records.append({"prompt": prompt, "minigpt": mini_text, "api": api_text})
    return records


def print_comparison(records: List[dict]) -> None:
    for r in records:
        print("=" * 70)
        print(f"Prompt : {r['prompt']}")
        print(f"MiniGPT: {r['minigpt']}")
        print(f"API    : {r['api'] if r['api'] is not None else '(API not configured)'}")


def configured_client() -> Optional[LLMClient]:
    """Return an LLMClient if CHATAVALAI_* environment variables are set, otherwise None."""
    try:
        return LLMClient(LLMConfig.from_env())
    except LLMError as e:
        print(f"API not configured ({e}); running without the API comparison.")
        return None


def main() -> None:
    """Train on the sample corpus (or API-generated documents) and compare with the API model."""
    client = configured_client()
    raw = generate_sample_corpus()
    if client is not None:
        try:
            raw = raw + generate_corpus_with_llm(client, ["machine learning", "the history of Persian poetry"], per_topic=2)
            print(f"Added API-generated documents. Total raw documents: {len(raw)}")
        except LLMError as e:
            print(f"Corpus generation via API failed, using sample corpus only: {e}")

    tokenizer, model, losses, docs = train_on_pipeline(raw, steps=600)
    print(f"\nLoss: {losses[0]:.4f} -> {losses[-1]:.4f} | documents used: {len(docs)}")
    prompts = ["Machine learning is", "The transformer"]
    print_comparison(compare_with_llm(tokenizer, model, prompts, client=client))


# ===== UNIT TESTS =====

class _FakeClient:
    """Stands in for LLMClient in tests: returns numbered paragraphs without network access."""
    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=0.7, max_tokens=512):
        self.calls += 1
        return (f"Paragraph {self.calls}: transformers process token sequences with attention "
                f"and feed forward layers so that each position learns from earlier positions. ") * 2


def test_generate_corpus_with_llm_counts_and_types():
    fake = _FakeClient()
    docs = generate_corpus_with_llm(fake, ["topic a", "topic b"], per_topic=2)
    assert len(docs) == 4 and fake.calls == 4
    assert all(isinstance(d, str) and d for d in docs)


def test_masked_loss_ignores_unsupervised_positions():
    torch.manual_seed(0)
    model = MiniGPT(vocab_size=20, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=8, ff_dim=32)
    inputs = torch.randint(0, 20, (2, 8))
    labels = torch.randint(0, 20, (2, 8))
    labels[:, 4:] = IGNORE_INDEX
    full = masked_loss(model, inputs, labels)
    manual = cross_entropy_loss(model(inputs)[labels != IGNORE_INDEX].unsqueeze(0),
                                labels[labels != IGNORE_INDEX].unsqueeze(0))
    assert torch.allclose(full, manual, atol=1e-6)


def test_train_on_pipeline_reduces_loss():
    _, _, losses, docs = train_on_pipeline(generate_sample_corpus(), block_size=64, merges=100,
                                           steps=300, batch_size=8, lr=3e-3, log_every=1000)
    head = sum(losses[:5]) / 5
    tail = sum(losses[-20:]) / 20   # single-batch losses are noisy, so compare averages
    assert tail < 0.5 * head, "Loss must drop when training on pipeline output."
    assert len(docs) >= 2


def test_compare_without_api_returns_minigpt_only():
    tokenizer, model, _, _ = train_on_pipeline(generate_sample_corpus(), block_size=32, merges=40,
                                               steps=5, batch_size=4, log_every=1000)
    records = compare_with_llm(tokenizer, model, ["Machine learning"], client=None, max_new_tokens=8)
    assert records[0]["api"] is None and isinstance(records[0]["minigpt"], str)


def run_all_tests() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()