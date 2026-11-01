import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple, Union
from tokenizer1 import BPETokenizer

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text: str) -> List[str]:
    """
    Split input text into initial word/symbol chunks using the GPT-2 regex pattern.

    Args:
        text (str): Raw input text string to pre-tokenize.

    Returns:
        List[str]: A list of string chunks matched by the pre-tokenization regex.
    """
    return [match.group() for match in GPT2_PATTERN.finditer(text)]

def apply_merge(byte_seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Replace consecutive occurrences of a specific pair of token IDs in a sequence with a new token ID.

    Args:
        byte_seq (List[int]): Current sequence of token IDs.
        pair (Tuple[int, int]): A tuple (first_id, second_id) representing the pair to merge.
        new_id (int): The new token ID assigned to the merged pair.

    Returns:
        List[int]: A new list of token IDs with target pairs merged.
    """
    result = []
    i = 0
    while i < len(byte_seq):
        if i + 1 < len(byte_seq) and (byte_seq[i], byte_seq[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(byte_seq[i])
            i += 1
    return result

class SpecialTokenHandler:
    """
    Manages registration and regex-based splitting of special tokens during tokenization.
    """

    def __init__(self) -> None:
        """Initialize empty special tokens mapping and pattern compiler."""
        self.special_tokens: Dict[str, int] = {}
        self.pattern: Union[re.Pattern, None] = None

    def add_token(self, token_str: str, token_id: int) -> None:
        """
        Register a special token and update the combined regular expression pattern.

        Args:
            token_str (str): The string representation of the special token (e.g., '<|end|>').
            token_id (int): The integer vocabulary ID assigned to the special token.

        Returns:
            None
        """
        self.special_tokens[token_str] = token_id
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self.pattern = re.compile("|".join(re.escape(k) for k in ordered))

    def split_with_specials(self, text: str) -> List[Tuple[str, bool]]:
        """
        Segment text into a list of tuples containing text chunks and boolean flags indicating special tokens.

        Args:
            text (str): Input text that may contain special tokens.

        Returns:
            List[Tuple[str, bool]]: A list of tuples where each tuple is (substring, is_special_flag).
        """
        if self.pattern is None:
            return [(text, False)] if text else []

        result: List[Tuple[str, bool]] = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                result.append((text[last_end:match.start()], False))
            result.append((match.group(), True))
            last_end = match.end()

        if last_end < len(text):
            result.append((text[last_end:], False))
        return result

class ProductionTokenizer(BPETokenizer):
    """
    Byte-Pair Encoding (BPE) tokenizer supporting training, normalization, special tokens, encoding, and decoding.
    """

    def __init__(self) -> None:
        """Initialize vocabulary, merges, special token handler, and next available token ID."""
        super().__init__()
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id: int = 256

    def normalize(self, text: str) -> str:
        """
        Normalize input text using Unicode NFKC normalization.

        Args:
            text (str): Raw input string.

        Returns:
            str: Unicode normalized string.
        """
        return unicodedata.normalize('NFKC' , text)

    def _to_byte_chunks(self, text: str) -> List[List[int]]:
        """Pre-tokenize plain text and convert each chunk to a list of byte values.

        Args:
            text (str): Text without special tokens.

        Returns:
            List[List[int]]: One byte sequence per chunk.
        """
        return [list(chunk.encode("utf-8")) for chunk in pre_tokenize(text)]
    
    def train(self, text: str, num_merges: int) -> None:
        """
        Train the BPE tokenizer by identifying frequent adjacent pairs and iteratively merging them.

        Args:
            text (str): Corpus text used for training the tokenizer.
            num_merges (int): Number of BPE merge operations to execute.

        Returns:
            None
        """
        if self.merges:
            raise RuntimeError("train() can only be called once on a fresh tokenizer.")
        
        text = self.normalize(text)
        byte_chunks = []
        for part, is_special in self.special_handler.split_with_specials(text):
            if not is_special:
                byte_chunks.extend(self._to_byte_chunks(part))

        for _ in range(num_merges):
            pair_counts = Counter(p for seq in byte_chunks for p in zip(seq, seq[1:]))
            if not pair_counts:
                break
            best_pair, _ = pair_counts.most_common(1)[0]

            new_id = self.next_id
            self.next_id += 1
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            byte_chunks = [self._merge_pair(seq, best_pair, new_id) for seq in byte_chunks]

    def add_special_token(self, token_str: str) -> int:
        """
        Register a new special token into the tokenizer vocabulary and special token handler.

        Args:
            token_str (str): Special token string (e.g., '<|begin|>').

        Returns:
            int: Assigned vocabulary integer ID for the special token.
        """
        new_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, new_id)
        self.vocab[new_id] = token_str.encode("utf-8")
        return new_id

    def encode(self, text: str) -> List[int]:
        """
        Encode raw text into a sequence of vocabulary token IDs using trained merges and special tokens.

        Args:
            text (str): Input text string to be tokenized.

        Returns:
            List[int]: List of encoded vocabulary token IDs.
        """
        text = self.normalize(text)
        ids: List[int] = []
        for part, is_special in self.special_handler.split_with_specials(text):
            if is_special:
                ids.append(self.special_handler.special_tokens[part])
                continue
            for seq in self._to_byte_chunks(part):
                for pair, new_id in self.merges.items():
                    seq = self._merge_pair(seq, pair, new_id)
                ids.extend(seq)
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a UTF-8 string.

        Args:
            ids (List[int]): List of integer token IDs.

        Returns:
            str: Decoded UTF-8 text string.
        """
        return b"".join(self.get_token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def get_token_bytes(self, token_id: int) -> bytes:
        """
        Retrieve underlying byte representation of a given token ID.

        Args:
            token_id (int): Token ID to look up.

        Returns:
            bytes: Byte sequence corresponding to token_id, or default placeholder if not found.
        """
        try:
            return self.vocab[token_id]
        except KeyError as e:
            raise KeyError(f"Unknown token id: {token_id}") from e

# [KEEP_IMPLEMENTATION]
def demo_byte_encoding() -> None:
    """
    Demonstrate byte-level UTF-8 encoding across various languages and character sets.

    Returns:
        None
    """
    print("=" * 60)
    print("Byte-Level Encoding")
    print("=" * 60)

    texts = [
        ("English", "hello"),
        ("Chinese", "你好"),
        ("Japanese", "こんにちは"),
        ("Emoji", "🔥🌍"),
        ("Mixed", "hello你好🔥"),
        ("Code", "def f(x):"),
    ]

    for label, text in texts:
        b = list(text.encode("utf-8"))
        print(f"{label:10s}: {len(text):2d} chars -> {len(b):2d} bytes -> {b[:16]}{'...' if len(b) > 16 else ''}")


# [KEEP_IMPLEMENTATION]
def demo_pre_tokenization() -> None:
    """
    Demonstrate GPT-2 regular expression pre-tokenization on different text formats.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("Pre-Tokenization (GPT-2 Regex)")
    print("=" * 60)

    texts = [
        "Hello, world! Don't stop.",
        "def train(model, data):",
        "The price is $3.14 per unit.",
        "  multiple   spaces   here  ",
    ]

    for text in texts:
        chunks = pre_tokenize(text)
        print(f"\n'{text}'")
        print(f"  -> {chunks}")


# [KEEP_IMPLEMENTATION]
def demo_full_tokenizer() -> None:
    """
    Demonstrate end-to-end BPE training, special token addition, encoding, and decoding.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("Training Production Tokenizer")
    print("=" * 60)

    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox runs through the forest. "
        "Machine learning models process natural language. "
        "Machine learning transforms how we build software. "
        "Deep learning models need large datasets to train. "
        "def train(model, data): return model.fit(data) "
        "def predict(model, x): return model(x) "
        "for i in range(100): print(i) "
    )

    tok = ProductionTokenizer()
    tok.train(corpus, num_merges=50)

    bos_id = tok.add_special_token("<|begin|>")
    eos_id = tok.add_special_token("<|end|>")
    user_id = tok.add_special_token("<|user|>")
    asst_id = tok.add_special_token("<|assistant|>")

    print(f"\nVocab size: {tok.vocab_size()}")
    print(f"Special tokens: <|begin|>={bos_id}, <|end|>={eos_id}, <|user|>={user_id}, <|assistant|>={asst_id}")

    print("\n" + "=" * 60)
    print("Encoding Tests")
    print("=" * 60)

    test_texts = [
        "The quick brown fox.",
        "你好世界 Hello World",
        "🔥🌍🚀",
        "def foo(x): return x + 1",
        "<|begin|><|user|>Hello<|end|>",
        "Machine learning is powerful.",
    ]

    for text in test_texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        raw_bytes = len(text.encode("utf-8"))
        print(f"\nInput:   {text}")
        print(f"IDs:     {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"Tokens:  {len(ids)} (from {raw_bytes} bytes, ratio: {len(ids)/raw_bytes:.2f})")
        print(f"Decoded: {decoded}")
        roundtrip = "PASS" if decoded == text else "FAIL"
        print(f"Round-trip: {roundtrip}")


# [KEEP_IMPLEMENTATION]
def demo_tiktoken_comparison() -> None:
    """
    Compare custom tokenizer performance and fertility against OpenAI's tiktoken.

    Returns:
        None
    """
    try:
        import tiktoken
    except ImportError:
        print("\ntiktoken not installed. Run: pip install tiktoken")
        return

    print("\n" + "=" * 60)
    print("Comparison with tiktoken (GPT-4)")
    print("=" * 60)

    enc = tiktoken.get_encoding("cl100k_base")

    test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

    tokens = enc.encode(test_paragraph)
    pieces = [enc.decode([t]) for t in tokens]

    print(f"\nInput: {test_paragraph}")
    print(f"GPT-4 tokens ({len(tokens)}): {pieces}")

    languages = [
        ("English", "The quick brown fox jumps over the lazy dog."),
        ("Chinese", "快速的棕色狐狸跳过了懒狗。"),
        ("Japanese", "素早い茶色のキツネが怠け者の犬を飛び越えた。"),
        ("Korean", "빠른 갈색 여우가 게으른 개를 뛰어넘었다."),
        ("Code", "def quicksort(arr): return sorted(arr)"),
        ("Emoji", "🎉🎊🎈🎁🎂🎄🎃🎆🎇✨"),
    ]

    print(f"\n{'Language':<10} {'Chars':<6} {'Tokens':<7} {'Fertility':<10}")
    print("-" * 35)
    for label, text in languages:
        toks = enc.encode(text)
        words = len(text.split())
        fertility = len(toks) / max(words, 1)
        print(f"{label:<10} {len(text):<6} {len(toks):<7} {fertility:<10.2f}")


# ===== UNIT TESTS =====
def test_pre_tokenize() -> None:
    """Test pre_tokenize output structure and edge cases."""
    res = pre_tokenize("Hello world! 123")
    assert isinstance(res, list), "Pre-tokenize output must be a list of strings."
    assert len(res) > 0, "Pre-tokenize output should not be empty for non-empty text."
    assert "".join(res) == "Hello world! 123", "Concatenated pre-tokenized chunks must reconstruct original text."

    # Edge Case: empty string
    empty_res = pre_tokenize("")
    assert empty_res == [], "Edge Case Failed: Empty string must return an empty list."


def test_apply_merge() -> None:
    """Test token pair merging logic and sequence lengths."""
    seq = [10, 20, 10, 20, 30]
    merged = apply_merge(seq, (10, 20), 100)
    assert isinstance(merged, list), "apply_merge must return a list."
    assert len(merged) == 3, f"Expected merged length of 3, got {len(merged)}. Check pair replacement step."
    assert merged == [100, 100, 30], "Merged output values do not match expected replaced token IDs."

    # Edge Case: single element sequence
    single = apply_merge([10], (10, 20), 100)
    assert single == [10], "Edge Case Failed: Single element list should remain unchanged."


def test_special_token_handler() -> None:
    """Test special token pattern creation and text splitting."""
    handler = SpecialTokenHandler()
    handler.add_token("<|end|>", 500)
    parts = handler.split_with_specials("Hello<|end|>World")

    assert isinstance(parts, list), "split_with_specials must return a list."
    assert len(parts) == 3, f"Expected 3 parts after splitting, got {len(parts)}."
    assert parts[1] == ("<|end|>", True), "Special token tag flag or value is incorrect."

    # Edge Case: text with no special tokens
    no_specials = handler.split_with_specials("Plain text")
    assert no_specials == [("Plain text", False)], "Edge Case Failed: Plain text splitting mismatched."


def test_production_tokenizer_train() -> None:
    """Test tokenizer training and vocabulary expansion."""
    tok = ProductionTokenizer()
    initial_vocab_size = tok.vocab_size()
    tok.train("abc abc abc", num_merges=2)

    assert tok.vocab_size() > initial_vocab_size, "Vocabulary size should increase after training BPE merges."
    assert len(tok.merges) <= 2, "Number of recorded merges should not exceed requested num_merges."

    # Edge Case: training on empty text
    tok_empty = ProductionTokenizer()
    tok_empty.train("", num_merges=5)
    assert tok_empty.vocab_size() == 256, "Edge Case Failed: Vocabulary size should stay at 256 for empty training text."


def test_production_tokenizer_encode_decode() -> None:
    """Test tokenizer round-trip integrity, output shape, and special tokens handling."""
    tok = ProductionTokenizer()
    tok.train("The quick brown fox jumps over the lazy dog.", num_merges=10)
    tok.add_special_token("<|special|>")

    text = "The quick fox <|special|>"
    encoded = tok.encode(text)

    assert isinstance(encoded, list), "Encoder output must be a list of integers."
    assert len(encoded) > 0, "Encoded token list should not be empty."
    assert all(isinstance(idx, int) for idx in encoded), "All token IDs in encoded list must be integers."

    decoded = tok.decode(encoded)
    assert isinstance(decoded, str), "Decoder output must be a string."
    assert decoded == text, f"Round-trip decoded text '{decoded}' does not match original text '{text}'."

    # Edge Case: single character string encoding
    single_char_ids = tok.encode("A")
    assert len(single_char_ids) == 1, "Edge Case Failed: Single byte character should yield exactly 1 token ID."


if __name__ == "__main__":
    test_pre_tokenize()
    test_apply_merge()
    test_special_token_handler()
    test_production_tokenizer_train()
    test_production_tokenizer_encode_decode()

    demo_byte_encoding()
    demo_pre_tokenization()
    demo_full_tokenizer()
    demo_tiktoken_comparison()