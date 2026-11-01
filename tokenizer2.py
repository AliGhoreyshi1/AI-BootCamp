import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple, Union

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
    # TODO: Apply GPT2_PATTERN regex iterator over text to extract all chunk string matches
    # matches = GPT2_PATTERN.finditer(text)
    # chunks = []
    # for match in matches:
    #     chunks.append(match.group())

    # return chunks

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
    # TODO: Iterate through byte_seq, find adjacent matching pairs, and replace them with new_id
    result = []

    i = 0

    while i < len(byte_seq):
        if (i + 1 < len(byte_seq) 
        and  byte_seq[i] == pair[0] 
        and byte_seq[i + 1] == pair[1]
            ):
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
        # TODO: Store the token ID mapping and update the compiled regex pattern using re.escape
        self.special_tokens[token_str] = token_id

        # for test 
        dict_keys = []
        # for key in self.special_tokens.keys():
        #     dict_keys.append(re.escape(key))

        pattern = '|'.join(re.escape(key) for key in self.special_tokens.keys())
        self.pattern = re.compile(pattern)
        
            

        

    def split_with_specials(self, text: str) -> List[Tuple[str, bool]]:
        """
        Segment text into a list of tuples containing text chunks and boolean flags indicating special tokens.

        Args:
            text (str): Input text that may contain special tokens.

        Returns:
            List[Tuple[str, bool]]: A list of tuples where each tuple is (substring, is_special_flag).
        """
        # TODO: Search text using pattern, split into standard text vs special token parts, and tag each part
        

        result = []

        if self.pattern is None:
            result.append((text , False))
            return result

        matches = self.pattern.finditer(text)
        last_end = 0

        for match in matches:
            normal_text = text[last_end :match.start()]

            if normal_text:
                result.append((normal_text , False))
            
            result.append((match.group() , True))
            last_end = match.end()

        remaining = text[last_end :]
        if remaining:
            result.append((remaining , False))

        return result



class ProductionTokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer supporting training, normalization, special tokens, encoding, and decoding.
    """

    def __init__(self) -> None:
        """Initialize vocabulary, merges, special token handler, and next available token ID."""
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_handler: SpecialTokenHandler = SpecialTokenHandler()
        self.next_id: int = 256

    def normalize(self, text: str) -> str:
        """
        Normalize input text using Unicode NFKC normalization.

        Args:
            text (str): Raw input string.

        Returns:
            str: Unicode normalized string.
        """
        # TODO: Normalize input text using unicodedata NFKC standard
        return unicodedata.normalize('NFKC' , text)

    def train(self, text: str, num_merges: int) -> None:
        """
        Train the BPE tokenizer by identifying frequent adjacent pairs and iteratively merging them.

        Args:
            text (str): Corpus text used for training the tokenizer.
            num_merges (int): Number of BPE merge operations to execute.

        Returns:
            None
        """
        # TODO: Normalize and pre-tokenize corpus text into byte sequences
        # TODO: Iteratively count adjacent pair frequencies across all chunk byte sequences
        # TODO: Find the most frequent pair, create a new vocabulary entry, and record the merge rule
        # TODO: Replace the best pair in all chunk sequences using apply_merge
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        byte_chunks = []
        for chunk in chunks:
            byte_chunks.append(list(chunk.encode('utf-8')))

        for _ in range(num_merges):
            pair_counts = {}

            for byte_seq in byte_chunks:
                for i in range(len(byte_seq) - 1):
                    pair = (byte_seq[i] , byte_seq[i+1])
                    if pair in pair_counts:
                        pair_counts[pair] += 1
                    else:
                        pair_counts[pair] = 1

            if not pair_counts:
                break
            best_pair = max(pair_counts , key=lambda x: pair_counts[x])

            next_id = self.next_id
            self.next_id += 1
            self.merges[best_pair] = next_id
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            for i , byte_seq in enumerate(byte_chunks):
                byte_chunks[i] = apply_merge(byte_seq , best_pair , next_id)



    def add_special_token(self, token_str: str) -> int:
        """
        Register a new special token into the tokenizer vocabulary and special token handler.

        Args:
            token_str (str): Special token string (e.g., '<|begin|>').

        Returns:
            int: Assigned vocabulary integer ID for the special token.
        """
        # TODO: Allocate new_id, register special token with special_handler and add byte representation to vocab
        new_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str , new_id)
        self.vocab[new_id] = token_str.encode('utf-8')

        return new_id

    
    def encode(self, text: str) -> List[int]:
        """
        Encode raw text into a sequence of vocabulary token IDs using trained merges and special tokens.

        Args:
            text (str): Input text string to be tokenized.

        Returns:
            List[int]: List of encoded vocabulary token IDs.
        """
        # TODO: Normalize input text and split into standard text and special token segments
        # TODO: For standard text segments, apply pre-tokenization and convert chunks into byte sequences
        # TODO: Apply learned BPE merges sequentially to byte sequences and collect all output token IDs
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        ids = []

        for part , is_special in parts:
            if is_special:
                ids.append(self.special_handler.special_tokens[part])
            else:
                chunks = pre_tokenize(part)
                byte_chunks = []
                for chunk in chunks:
                    byte_chunks.append(list(chunk.encode('utf-8')))
                for best_pair , new_id in self.merges.items():
                    for i , byte_seq in enumerate(byte_chunks):
                        byte_chunks[i] = apply_merge(byte_seq , best_pair , new_id)

                for byte_seq in byte_chunks:
                    ids.extend(byte_seq)

        return ids
            

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a UTF-8 string.

        Args:
            ids (List[int]): List of integer token IDs.

        Returns:
            str: Decoded UTF-8 text string.
        """
        # TODO: Map token IDs back to byte representations using vocabulary and decode as UTF-8
        decoded = b''.join(self.get_token_bytes(token_id) for token_id in ids)
        return decoded.decode('utf-8')

    def vocab_size(self) -> int:
        """
        Get current total size of vocabulary including base bytes, merges, and special tokens.

        Returns:
            int: Number of total entries in vocabulary.
        """
        # TODO: Return total number of vocabulary items
        return len(self.vocab)

    def get_token_bytes(self, token_id: int) -> bytes:
        """
        Retrieve underlying byte representation of a given token ID.

        Args:
            token_id (int): Token ID to look up.

        Returns:
            bytes: Byte sequence corresponding to token_id, or default placeholder if not found.
        """
        # TODO: Retrieve byte mapping from vocabulary dictionary for specified token_id
        return self.vocab.get(token_id , b"?")


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
    # demo_byte_encoding()
    # demo_pre_tokenization()
    # demo_full_tokenizer()
    # demo_tiktoken_comparison()
    
    test_pre_tokenize()
    print("test_pre_tokenize PASSED")

    test_apply_merge()
    print("test_apply_merge PASSED")

    test_special_token_handler()
    print("test_special_token_handler PASSED")

    test_production_tokenizer_train()
    print("test_production_tokenizer_train PASSED")

    test_production_tokenizer_encode_decode()
    print("test_production_tokenizer_encode_decode PASSED")