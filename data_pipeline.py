import re
import hashlib
import random
import time
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple, Set, Dict, Any, Generator, Optional
from tokenizer2 import ProductionTokenizer

def clean_text(text: str) -> str:
    """
    Cleans raw document text by stripping HTML tags and URLs, and normalizing whitespace.

    Non-ASCII characters are kept on purpose: the pipeline is multilingual, and
    Unicode normalization (NFKC) is applied by the tokenizer, not here.

    Args:
        text (str): Input raw text document.

    Returns:
        str: Cleaned and normalized text string.
    """
    text = re.sub(r"(?:https?://|www\.)[^\s<>\"']+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def quality_filter(
    text: str, 
    min_words: int = 50, 
    max_ratio_caps: int = 0.3, 
    max_ratio_special: float = 0.1
) -> bool:
    """
        Filters out low-quality documents based on length, capitalization ratio, and special character density.
    
        Length is counted with _units(), so scripts without spaces (CJK) are measured
        in characters rather than being treated as one giant word.
    
        Args:
            text (str): Cleaned document text.
            min_words (int): Minimum required unit count (words, or CJK characters).
            max_ratio_caps (float): Maximum allowed ratio of ALL-CAPS words.
            max_ratio_special (float): Maximum allowed ratio of non-alphanumeric special characters.
    
        Returns:
            bool: True if the document meets quality criteria, False otherwise.
        """
    words = text.split()
    if not words or len(_units(text)) < min_words:
        return False

    caps_ratio = sum(word.isupper() for word in words) / len(words)
    if caps_ratio > max_ratio_caps:
        return False

    special_count = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    special_ratio = special_count / len(text) if text else 0.0
    return special_ratio <= max_ratio_special

# CJK scripts (kana, ideographs, hangul) are written without spaces, so each
# character is treated as its own unit. Other scripts use whitespace-free word runs.
_CJK_RANGES = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
_UNIT_PATTERN = re.compile(rf"[{_CJK_RANGES}]|[^\W{_CJK_RANGES}]+")

def _units(text: str) -> List[str]:
    """
    Split text into lowercase units: word runs for most scripts, single characters for CJK.

    Args:
        text (str): Input text.

    Returns:
        List[str]: Lowercased units in order of appearance.
    """
    return _UNIT_PATTERN.findall(text.lower())

def get_shingles(text: str, k: int = 5) -> Set[str]:
    """
    Extracts k-shingles (word n-grams) from a text string.

    Args:
        text (str): Input document text.
        k (int): Size of word shingle (n-gram length).

    Returns:
        Set[str]: Unique set of k-word shingles.
    """
    if k<=0:
        raise ValueError("k must be positive")
    words = re.findall(r"\b\w+\b", text.lower())
    return {" ".join(words[i:i+k]) for i in range(len(words)- k + 1)}

def minhash_signature(shingles: Set[str], num_hashes: int = 128) -> List[int]:
    """
    Computes MinHash signature for a set of shingles using hash function permutations.

    Args:
        shingles (Set[str]): Set of unique word shingles.
        num_hashes (int): Length of the MinHash signature vector.

    Returns:
        List[int]: MinHash signature list of length (num_hashes,).
    """
    if num_hashes<0:
        raise ValueError("num_hashes must be nonnegetive")
    if not shingles:
        return [0]*num_hashes

    signature = []
    encoded_shingles = [shingle.encode("utf-8") for shingle in shingles]
    for seed in range(num_hashes):
        prefix = seed.to_bytes(8, byteorder="big")
        minimum = min(
            int.from_bytes(hashlib.sha256(prefix+shingle).digest()[:8], "big")
            for shingle in encoded_shingles
        )
        signature.append(minimum)
    return signature

# shingles = {"the cat", "cat is", "is cute"}
# signature = minhash_signature(shingles, num_hashes=4)
# print(signature)

def lsh_buckets(signature: List[int], bands: int = 16) -> List[Tuple[int, str]]:
    """
    Divides MinHash signature into bands and maps each band to a bucket identifier using LSH.

    Args:
        signature (List[int]): MinHash signature vector of shape (num_hashes,).
        bands (int): Number of bands to divide the signature into.

    Returns:
        List[Tuple[int, str]]: List of (band_id, bucket_hash) tuples of length (bands,).
    """
    if bands <= 0:
            raise ValueError("bands must be a positive integer")
    if len(signature) < bands:
        raise ValueError("signature must contain at least one value per band")

    buckets = []
    for band_id in range(bands):
        start = band_id * len(signature) // bands
        end = (band_id + 1) * len(signature) // bands
        chunk = signature[start:end]
        chunk_bytes = ",".join(str(value) for value in chunk).encode("ascii")
        buckets.append((band_id, hashlib.sha256(chunk_bytes).hexdigest()))
    return buckets

# buckets = lsh_buckets(signature, bands=2)

def deduplicate(
    documents: List[str], 
    threshold: float = 0.8, 
    num_hashes: int = 128, 
    bands: int = 16
) -> Tuple[List[str], int]:
    """
    Finds and removes near-duplicate documents using MinHash LSH and Jaccard similarity evaluation.

    Args:
        documents (List[str]): List of clean document strings.
        threshold (float): Jaccard similarity threshold for considering two docs as duplicates.
        num_hashes (int): Total hash functions for MinHash computation.
        bands (int): Number of bands for LSH partitioning.

    Returns:
        Tuple[List[str], int]: Tuple containing (list of deduplicated documents, count of removed duplicates).
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if num_hashes < bands or bands <= 0:
        raise ValueError("num_hashes must be >= bands > 0")
    if len(documents) < 2:
        return list(documents), 0

    shingle_sets = [get_shingles(document, k=5) for document in documents]
    signatures = [minhash_signature(shingle, num_hashes) for shingle in shingle_sets]

    buckets = defaultdict(list)
    for doc_id, signature in enumerate(signatures):
        for bucket in lsh_buckets(signature, bands):
            buckets[bucket].append(doc_id)

    candidate_pairs = set()
    for indices in buckets.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1:]:
                candidate_pairs.add((left, right))  

    buckets = defaultdict(list)
    for doc_id, signature in enumerate(signatures):
        for bucket in lsh_buckets(signature, bands):
            buckets[bucket].append(doc_id)

    candidate_pairs = set()
    for indices in buckets.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1:]:
                candidate_pairs.add((left, right))

    removed = set()
    # Keep the earliest occurrence of each duplicate, retaining original order.
    for left, right in sorted(candidate_pairs):
        if left in removed or right in removed:
            continue
        union = shingle_sets[left] | shingle_sets[right]
        similarity = (len(shingle_sets[left] & shingle_sets[right]) / len(union)
                      if union else float(documents[left] == documents[right]))
        if similarity >= threshold:
            removed.add(right)

    deduplicated = [doc for i, doc in enumerate(documents) if i not in removed]
    return deduplicated, len(removed)

# documents = [
#     "machine learning is useful for artificial intelligence",
#     "machine learning is useful for artificial intelligence",
#     "computer vision works with digital images"
# ]
# print(deduplicate(documents)[0])

class SimpleTokenizer:
    """
    A minimal Byte Pair Encoding (BPE) tokenizer implementation operating over byte sequences.
    """

    EOS_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"

    def __init__(self, vocab_size: int = 256):
        """
        Initializes tokenizer vocabulary and internal mapping structures.

        Args:
            vocab_size (int): Target vocabulary capacity.
        """
        self._tok = ProductionTokenizer()
        self.eos_id: int = self._tok.add_special_token(self.EOS_TOKEN)
        self.pad_id: int = self._tok.add_special_token(self.PAD_TOKEN)
        self._special_ids = {self.eos_id, self.pad_id}

    def train_bpe(self, text: str, num_merges: int) -> None:
        """
        Iteratively finds the most frequent pair of tokens and merges them into a new token.

        Args:
            text (str): Training text corpus.
            num_merges (int): Number of BPE merge operations to execute.

        Returns:
            None
        """
        self._tok.train(text, num_merges)

    def encode(self, text: str) -> List[int]:
        """
        Encodes input text into a list of token IDs using pre-learned BPE merge rules.

        Args:
            text (str): Input text string.

        Returns:
            List[int]: Encoded list of token IDs.
        """
        return self._tok.encode(text)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a UTF-8 string.

        Args:
            ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text representation.
        """
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_ids]
        return self._tok.decode(ids)

    def vocab_size(self) -> int:
        """
        Returns the total vocabulary size including base bytes and merges.

        Returns:
            int: Total vocabulary size.
        """
        return self._tok.vocab_size()


def tokenize_corpus(documents: List[str], tokenizer: SimpleTokenizer) -> List[int]:
    """
    Tokenizes a list of documents into a single flat stream of token IDs separated by EOS markers.

    Args:
        documents (List[str]): List of document strings.
        tokenizer (SimpleTokenizer): Trained BPE tokenizer instance.

    Returns:
        List[int]: Flat list containing combined token sequence.
    """
    if tokenizer.eos_id is None:
        raise ValueError("tokenizer must define an eos_id")
    token_ids: List[int] = []
    for doc in documents:
        token_ids.extend(tokenizer.encode(doc))
        token_ids.append(tokenizer.eos_id)
    return token_ids


def pack_sequences(
    token_ids: List[int], 
    seq_length: int, 
    pad_id: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Packs continuous token stream into fixed-length sequence chunks with corresponding attention masks.

    Args:
        token_ids (List[int]): Continuous flat list of token IDs.
        seq_length (int): Target length per sequence block.
        pad_id (int): Token ID used for padding incomplete trailing sequences.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: Tuple of (padded_sequences, attention_masks)
            where each element has dynamic batch outer dimension and sequence shape (seq_length,).
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    sequences: List[List[int]] = []
    masks: List[List[int]] = []
    for start in range(0, len(token_ids), seq_length):
        chunk = token_ids[start:start + seq_length]
        padding = seq_length - len(chunk)
        sequences.append(chunk + [pad_id] * padding)
        masks.append([1] * len(chunk) + [0] * padding)
    return sequences, masks


class PreTrainingDataLoader:
    """
    Data loader providing mini-batch iteration and shuffling over packed token sequences.
    """
    def __init__(
        self, 
        sequences: List[List[int]], 
        attention_masks: List[List[int]], 
        batch_size: int, 
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        """
        Initializes dataloader properties.

        Args:
            sequences (List[List[int]]): Packed input sequences of shape (num_samples, seq_length).
            attention_masks (List[List[int]]): Attention masks of shape (num_samples, seq_length).
            batch_size (int): Size of individual mini-batches.
            shuffle (bool): Whether to shuffle sample indices prior to batching.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("require 0 <= rank < world_size")
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
            """Set the epoch number so that each epoch gets a different, reproducible shuffle."""
            self.epoch = epoch

    def _indices(self) -> List[int]:
        indices = list(range(self.rank, len(self.sequences), self.world_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        return indices

    def __len__(self) -> int:
        """
        Calculates total number of batches per epoch.

        Returns:
            int: Number of available mini-batches.
        """
        n = len(range(self.rank, len(self.sequences), self.world_size))
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Generator[Tuple[List[List[int]], List[List[int]]], None, None]:
        """
        Iterates over the dataset and yields mini-batches of sequences and attention masks.

        Returns:
            Generator[Tuple[List[List[int]], List[List[int]]], None, None]: Batch generator yielding 
            (batch_sequences, batch_attention_masks) where each tensor component has shape (batch_size, seq_length).
        """
        indices = self._indices()
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield ([self.sequences[i] for i in batch],
                   [self.attention_masks[i] for i in batch])

# ---------------------------------------------------------------------------
# Language detection (script-based heuristic)
# ---------------------------------------------------------------------------

PERSIAN_ONLY_LETTERS = set("پچژگکی")  # letters that Arabic script does not use
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

def detect_language(text: str) -> Tuple[str, float]:
    """
    Guess the dominant script/language of a text from the Unicode blocks of its letters.

    Latin-script text is labelled 'latin' (English and other Latin languages are not
    separated by this heuristic). Persian is separated from Arabic by its letters.

    Args:
        text (str): Input text.

    Returns:
        Tuple[str, float]: (language label, share of letters in that label). ('und', 0.0) if no letters.
    """
    counts: Counter = Counter()
    persian_letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            counts["ja"] += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            counts["zh"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["ko"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["ar"] += 1
            if ch in PERSIAN_ONLY_LETTERS:
                persian_letters += 1
        elif cp < 0x0250:  # ASCII and Latin-1/Latin Extended
            counts["latin"] += 1
        else:
            counts["other"] += 1

    if persian_letters > 0 and counts["ar"] > 0:
        counts["fa"] = counts.pop("ar")
    total = sum(counts.values())
    if total == 0:
        return "und", 0.0
    label, top = counts.most_common(1)[0]
    return label, top / total


def language_filter(
    text: str,
    allowed: Tuple[str, ...] = ("latin", "fa", "ar", "zh", "ja", "ko"),
    min_confidence: float = 0.7
) -> bool:
    """
    Keep a document only if its dominant language is allowed and sufficiently dominant.

    Args:
        text (str): Cleaned document text.
        allowed (Tuple[str, ...]): Accepted language labels from detect_language.
        min_confidence (float): Minimum share of letters that must belong to the dominant label.

    Returns:
        bool: True if the document is kept.
    """
    label, confidence = detect_language(text)
    return label in allowed and confidence >= min_confidence

# ---------------------------------------------------------------------------
# Repetition filter and PII removal
# ---------------------------------------------------------------------------

def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    """
    Fraction of unit n-grams that are repeats of an earlier n-gram (0 = no repetition).

    Args:
        text (str): Input text.
        n (int): n-gram size in units.

    Returns:
        float: Repetition ratio in [0, 1].
    """
    units = _units(text)
    grams = [tuple(units[i:i + n]) for i in range(len(units) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def is_repetitive(text: str, max_ratio: float = 0.5, n: int = 3) -> bool:
    """
    Flag documents dominated by repeated n-grams (typical of spam and keyword stuffing).

    Args:
        text (str): Cleaned document text.
        max_ratio (float): Maximum tolerated repetition ratio.
        n (int): n-gram size in units.

    Returns:
        bool: True if the document should be removed.
    """
    return repeated_ngram_ratio(text, n) > max_ratio


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NATIONAL_ID_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)")


def _replace_spans(text: str, pattern: "re.Pattern", token: str) -> str:
    """Replace regex matches in `text` with `token`, matching on a digit-normalized copy."""
    normalized = text.translate(_PERSIAN_DIGITS)  # 1:1 mapping, so spans line up with `text`
    out, last = [], 0
    for m in pattern.finditer(normalized):
        out.append(text[last:m.start()])
        out.append(token)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def remove_pii(text: str) -> str:
    """
    Replace e-mail addresses, national ID numbers and phone numbers with placeholders.
    Also strips literal special-token strings so that data cannot inject EOS/PAD ids.

    Args:
        text (str): Document text.

    Returns:
        str: Text with PII replaced by <EMAIL>, <ID> and <PHONE>.
    """
    text = text.replace("<|endoftext|>", " ").replace("<|pad|>", " ")
    text = _replace_spans(text, _EMAIL_RE, "<EMAIL>")
    text = _replace_spans(text, _NATIONAL_ID_RE, "<ID>")
    text = _replace_spans(text, _PHONE_RE, "<PHONE>")
    return text

# ---------------------------------------------------------------------------
# Streaming stages
# ---------------------------------------------------------------------------

def stream_clean(raw_docs: Iterable[str], min_confidence: float = 0.7) -> Generator[str, None, None]:
    """
    Lazily apply HTML/URL cleaning, language filtering and quality filtering.

    Args:
        raw_docs (Iterable[str]): Raw documents (any iterable, e.g. a file reader).
        min_confidence (float): Passed to language_filter.

    Yields:
        str: Cleaned documents that passed the filters.
    """
    for raw in raw_docs:
        doc = clean_text(raw)
        if not language_filter(doc, min_confidence=min_confidence):
            continue
        if not quality_filter(doc) or is_repetitive(doc):
            continue
        yield doc


def stream_dedup(docs: Iterable[str], batch_size: int = 1000, threshold: float = 0.8) -> Generator[str, None, None]:
    """
    Deduplicate a stream in fixed-size batches.

    Limitation: duplicates across batches are not detected. A full-corpus deduplication
    needs a sharded index (for example, one LSH index per shard, merged in a second pass).

    Args:
        docs (Iterable[str]): Cleaned documents.
        batch_size (int): Number of documents held in memory at once.
        threshold (float): Jaccard threshold passed to deduplicate.

    Yields:
        str: Documents that survived batch-level deduplication.
    """
    batch: List[str] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) == batch_size:
            kept, _ = deduplicate(batch, threshold=threshold)
            yield from kept
            batch = []
    if batch:
        kept, _ = deduplicate(batch, threshold=threshold)
        yield from kept


# ---------------------------------------------------------------------------
# Packing with document boundaries and next-token labels
# ---------------------------------------------------------------------------

def pack_documents(
    token_docs: List[List[int]],
    eos_id: int,
    seq_length: int,
    pad_id: int,
    ignore_index: int = -100,
) -> Dict[str, List[List[int]]]:
    """
    Pack tokenized documents into fixed-length blocks with per-token document ids.

    Every document ends with EOS. Labels are next-token targets: a position is supervised
    only when the next token belongs to the same document, so no loss is computed on
    predicting the first token of an unrelated document. Padding has label `ignore_index`.
    doc_ids are used by the model to build a block-diagonal attention mask.

    Args:
        token_docs (List[List[int]]): Token ids per document (without EOS).
        eos_id (int): End-of-sequence token id appended to each document.
        seq_length (int): Block length.
        pad_id (int): Padding token id.
        ignore_index (int): Label value that is excluded from the loss.

    Returns:
        Dict[str, List[List[int]]]: keys 'input_ids', 'labels', 'doc_ids', 'attention_mask'.
    """
    if seq_length < 2:
        raise ValueError("seq_length must be at least 2")

    flat_tokens: List[int] = []
    flat_docs: List[int] = []
    for doc_idx, doc in enumerate(token_docs):
        for t in doc + [eos_id]:
            flat_tokens.append(t)
            flat_docs.append(doc_idx)

    out: Dict[str, List[List[int]]] = {"input_ids": [], "labels": [], "doc_ids": [], "attention_mask": []}
    for start in range(0, len(flat_tokens), seq_length):
        toks = flat_tokens[start:start + seq_length]
        docs = flat_docs[start:start + seq_length]
        n = len(toks)
        labels = []
        for i in range(n):
            if i + 1 < n and docs[i + 1] == docs[i]:
                labels.append(toks[i + 1])
            else:
                labels.append(ignore_index)
        pad = seq_length - n
        out["input_ids"].append(toks + [pad_id] * pad)
        out["labels"].append(labels + [ignore_index] * pad)
        out["doc_ids"].append(docs + [-1] * pad)
        out["attention_mask"].append([1] * n + [0] * pad)
    return out


def build_training_data(
    raw_docs: Iterable[str],
    num_merges: int = 100,
    seq_length: int = 128,
    min_confidence: float = 0.7,
) -> Tuple["SimpleTokenizer", Dict[str, List[List[int]]], List[str]]:
    """
    Run the full pipeline and return a trained tokenizer and packed training blocks.

    Order follows the course diagram: clean -> language -> quality -> dedup -> PII -> tokenize -> pack.

    Args:
        raw_docs (Iterable[str]): Raw documents.
        num_merges (int): BPE merges to learn.
        seq_length (int): Block length for packing.
        min_confidence (float): Language confidence threshold.

    Returns:
        Tuple[SimpleTokenizer, Dict[str, List[List[int]]], List[str]]: (tokenizer, packed blocks, final documents).
    """
    filtered = stream_clean(raw_docs, min_confidence=min_confidence)
    deduped = list(stream_dedup(filtered))
    final_docs = [remove_pii(doc) for doc in deduped]
    if not final_docs:
        raise ValueError("no documents survived the pipeline")

    tokenizer = SimpleTokenizer()
    tokenizer.train_bpe(" ".join(final_docs), num_merges=num_merges)
    token_docs = [tokenizer.encode(doc) for doc in final_docs]
    packed = pack_documents(token_docs, tokenizer.eos_id, seq_length, tokenizer.pad_id)
    return tokenizer, packed, final_docs

def compute_statistics(
    documents: List[str],
    token_ids: List[int],
    sequences: List[List[int]],
    tokenizer_vocab_size: int,
    pad_id: int = 0
) -> Dict[str, Any]:
    """
    Computes overall corpus, vocabulary, compression, and sequence padding efficiency metrics.

    Args:
        documents (List[str]): Corpus of cleaned documents.
        token_ids (List[int]): Continuous flat list of token IDs.
        sequences (List[List[int]]): Packed token sequences.
        tokenizer_vocab_size (int): Size of the tokenizer vocabulary.

    Returns:
        Dict[str, Any]: Dictionary containing pre-training dataset analytical statistics.
    """
    total_chars = sum(len(doc) for doc in documents)
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))

    slots = sum(len(seq) for seq in sequences)
    non_pad = sum(1 for seq in sequences for token in seq if token != pad_id)

    doc_word_counts = [len(_units(doc)) for doc in documents]

    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size if tokenizer_vocab_size else 0.0,
        "compression_ratio": total_bytes / total_tokens if total_tokens else 0.0,
        "avg_doc_length_words": sum(doc_word_counts) / len(doc_word_counts) if doc_word_counts else 0.0,
        "num_sequences": len(sequences),
        "sequence_utilization": non_pad / slots if slots else 0.0,
        "top_tokens": Counter(token_ids).most_common(10),
    }

# [KEEP_IMPLEMENTATION]
def generate_sample_corpus() -> List[str]:
    """
    Generates a pre-defined synthetic corpus containing standard, duplicate, HTML, and spam documents.

    Returns:
        List[str]: List of raw document strings.
    """
    base_docs = [
        "Machine learning is a subset of artificial intelligence that provides systems the ability "
        "to automatically learn and improve from experience without being explicitly programmed. "
        "Machine learning focuses on the development of computer programs that can access data and "
        "use it to learn for themselves. The process of learning begins with observations or data, "
        "such as examples, direct experience, or instruction, in order to look for patterns in data "
        "and make better decisions in the future based on the examples that we provide.",

        "Deep learning is part of a broader family of machine learning methods based on artificial "
        "neural networks with representation learning. Learning can be supervised, semi-supervised "
        "or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural "
        "networks, convolutional neural networks and transformers have been applied to fields "
        "including natural language processing, speech recognition, computer vision, and many other tasks.",

        "Natural language processing is a subfield of linguistics, computer science, and artificial "
        "intelligence concerned with the interactions between computers and human language, in "
        "particular how to program computers to process and analyze large amounts of natural language "
        "data. The result is a computer capable of understanding the contents of documents, including "
        "the contextual nuances of the language within them.",

        "Transformers are a type of neural network architecture that has become the dominant approach "
        "for natural language processing tasks. The key innovation is the self-attention mechanism, "
        "which allows the model to weigh the importance of different parts of the input when producing "
        "each part of the output. This enables transformers to capture long-range dependencies in text "
        "much more effectively than previous recurrent approaches.",

        "The attention mechanism in neural networks allows the model to focus on relevant parts of "
        "the input sequence when generating each element of the output. In the transformer architecture, "
        "multi-head attention computes attention in parallel across multiple representation subspaces, "
        "enabling the model to jointly attend to information from different representation subspaces "
        "at different positions in the sequence.",

        "Reinforcement learning is an area of machine learning concerned with how intelligent agents "
        "ought to take actions in an environment in order to maximize the notion of cumulative reward. "
        "Reinforcement learning is one of three basic machine learning paradigms, alongside supervised "
        "learning and unsupervised learning. It differs from supervised learning in that correct input "
        "and output pairs need not be presented.",

        "Computer vision is an interdisciplinary scientific field that deals with how computers can "
        "gain high-level understanding from digital images or videos. From the perspective of "
        "engineering, it seeks to understand and automate tasks that the human visual system can do. "
        "Computer vision tasks include methods for acquiring, processing, analyzing and understanding "
        "digital images, and extraction of high-dimensional data from the real world.",

        "Convolutional neural networks are a class of deep learning architecture commonly applied to "
        "analyze visual imagery. They use a variation of multilayer perceptrons designed to require "
        "minimal preprocessing. They are also known as shift invariant or space invariant artificial "
        "neural networks based on their shared-weights architecture and translation invariance "
        "characteristics. Convolutional networks were inspired by biological processes.",

        "Generative adversarial networks consist of two neural networks that contest with each other "
        "in the form of a zero-sum game, where one agent gain is another agent loss. Given a training "
        "set, this technique learns to generate new data with the same statistics as the training set. "
        "For example, a generative adversarial network trained on photographs can generate new "
        "photographs that look authentic to human observers.",

        "Transfer learning is a machine learning method where a model developed for a task is reused "
        "as the starting point for a model on a second task. It is a popular approach in deep learning "
        "where pre-trained models are used as the starting point on computer vision and natural language "
        "processing tasks given the vast compute and time resources required to develop neural network "
        "models on these problems and the large improvements they provide.",
    ]

    near_dup_1 = (
        "Machine learning is a subset of artificial intelligence that provides systems the ability "
        "to automatically learn and improve from experience. Machine learning focuses on developing "
        "computer programs that can access data and use it to learn for themselves. The learning "
        "process begins with observations or data, such as examples or direct experience, in order "
        "to look for patterns and make better decisions based on the examples provided."
    )

    near_dup_2 = (
        "Deep learning is part of a broader family of machine learning methods based on artificial "
        "neural networks with representation learning. Learning can be supervised, semi-supervised "
        "or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural "
        "networks, convolutional neural networks and transformers have been applied to fields "
        "including natural language processing, speech recognition, computer vision, and many other tasks."
    )

    short_doc = "This is too short to be useful."

    html_doc = (
        "<html><body><h1>Title</h1><p>Machine learning is transforming how we build software. "
        "Deep neural networks can learn complex patterns from data. The transformer architecture "
        "has become the dominant approach for language tasks. Self-attention allows models to capture "
        "long-range dependencies. Pre-training on large corpora produces strong foundation models. "
        "Fine-tuning adapts these models to specific tasks with minimal additional data.</p></body></html>"
    )

    spam_doc = "BUY NOW CLICK HERE FREE MONEY GUARANTEED RESULTS " * 20

    docs = base_docs + [near_dup_1, near_dup_2, short_doc, html_doc, spam_doc]
    return docs

# [KEEP_IMPLEMENTATION]
def run_pipeline():
    """
    Executes end-to-end data processing pipeline for pre-training dataset preparation.
    """
    print("=" * 60)
    print("Data Pipeline for Pre-Training")
    print("=" * 60)

    raw_docs = generate_sample_corpus()
    print(f"\nRaw documents: {len(raw_docs)}")

    print("\n--- Stage 1: Cleaning ---")
    cleaned_docs = [clean_text(doc) for doc in raw_docs]
    print(f"After HTML stripping: {len(cleaned_docs)} documents")

    print("\n--- Stage 2: Quality Filtering ---")
    filtered_docs = [doc for doc in cleaned_docs if quality_filter(doc)]
    removed_quality = len(cleaned_docs) - len(filtered_docs)
    print(f"Removed {removed_quality} low-quality documents")
    print(f"Remaining: {len(filtered_docs)} documents")

    print("\n--- Stage 3: Deduplication (MinHash + LSH) ---")
    start = time.time()
    deduped_docs, num_removed = deduplicate(filtered_docs, threshold=0.8)
    dedup_time = time.time() - start
    print(f"Removed {num_removed} near-duplicates in {dedup_time:.2f}s")
    print(f"Remaining: {len(deduped_docs)} documents")

    print("\n--- Stage 4: Tokenization ---")
    all_text = " ".join(deduped_docs)
    tokenizer = SimpleTokenizer()
    start = time.time()
    tokenizer.train_bpe(all_text, num_merges=100)
    train_time = time.time() - start
    print(f"Trained tokenizer with {tokenizer.vocab_size()} tokens in {train_time:.2f}s")

    start = time.time()
    token_ids = tokenize_corpus(deduped_docs, tokenizer)
    tok_time = time.time() - start
    print(f"Tokenized {len(token_ids):,} tokens in {tok_time:.2f}s ({len(token_ids)/max(tok_time, 0.001):,.0f} tokens/sec)")

    print("\n--- Stage 5: Sequence Packing ---")
    seq_length = 128
    sequences, masks = pack_sequences(token_ids, seq_length, pad_id=0)
    print(f"Packed into {len(sequences)} sequences of length {seq_length}")

    print("\n--- Stage 6: DataLoader ---")
    batch_size = 4
    loader = PreTrainingDataLoader(sequences, masks, batch_size)
    print(f"DataLoader: {len(loader)} batches of size {batch_size}")

    batch_count = 0
    total_tokens_served = 0
    for batch_seqs, batch_masks in loader:
        batch_count += 1
        total_tokens_served += sum(sum(m) for m in batch_masks)
        if batch_count <= 2:
            print(f"\n  Batch {batch_count}:")
            print(f"    Sequences: {len(batch_seqs)}")
            print(f"    First seq (first 20 tokens): {batch_seqs[0][:20]}...")
            print(f"    First mask (first 20): {batch_masks[0][:20]}...")
    print(f"\n  Total batches served: {batch_count}")
    print(f"  Total non-padding tokens served: {total_tokens_served:,}")

    print("\n--- Dataset Statistics ---")
    stats = compute_statistics(deduped_docs, token_ids, sequences, tokenizer.vocab_size())
    print(f"  Documents:           {stats['total_documents']}")
    print(f"  Total characters:    {stats['total_characters']:,}")
    print(f"  Total tokens:        {stats['total_tokens']:,}")
    print(f"  Unique tokens:       {stats['unique_tokens']}")
    print(f"  Vocab utilization:  {stats['vocab_utilization']:.1%}")
    print(f"  Compression ratio:  {stats['compression_ratio']:.2f} chars/token")
    print(f"  Avg doc length:     {stats['avg_doc_length_words']:.0f} words")
    print(f"  Num sequences:      {stats['num_sequences']}")
    print(f"  Seq utilization:    {stats['sequence_utilization']:.1%}")

    print("\n--- Pipeline Summary ---")
    print(f"  Raw documents:       {len(raw_docs)}")
    print(f"  After cleaning:      {len(cleaned_docs)}")
    print(f"  After quality filter: {len(filtered_docs)} (-{removed_quality})")
    print(f"  After dedup:         {len(deduped_docs)} (-{num_removed})")
    print(f"  Final tokens:        {len(token_ids):,}")
    print(f"  Training sequences:  {len(sequences)}")
    print(f"  Training batches:    {len(loader)}")


# ===== UNIT TESTS =====
def test_clean_text():
    sample = "<html><body>Hello    World! http://example.com</body></html>"
    cleaned = clean_text(sample)
    assert isinstance(cleaned, str), "Output must be a string."
    assert "<html>" not in cleaned, "HTML tags were not removed."
    assert "http" not in cleaned, "URLs were not stripped."
    assert "  " not in cleaned, "Multiple spaces were not collapsed."
    
    # Edge case: Empty input string
    assert clean_text("") == "", "Edge case failed: Empty string must return an empty string."

def test_quality_filter():
    good_text = " ".join(["word"] * 60)
    bad_text_short = "short text"
    bad_text_caps = " ".join(["WORD"] * 60)
    
    assert quality_filter(good_text) is True, "Valid document failed quality filter."
    assert quality_filter(bad_text_short) is False, "Short document incorrectly passed quality filter."
    assert quality_filter(bad_text_caps) is False, "Overly-capitalized document passed quality filter."
    
    # Edge case: Boundary word count
    exact_words = " ".join(["word"] * 50)
    assert quality_filter(exact_words, min_words=50) is True, "Edge case failed: Exact min_words boundary missed."

def test_get_shingles():
    text = "the quick brown fox jumps over the lazy dog"
    shingles = get_shingles(text, k=3)
    assert isinstance(shingles, set), "Output should be a set."
    assert len(shingles) == 7, "Shingle extraction length mismatch."
    
    # Edge case: Text shorter than k
    short_text = "quick brown"
    assert get_shingles(short_text, k=5) == set(), "Edge case failed: Text shorter than k must return an empty set."

def test_minhash_signature():
    shingles = {"quick brown fox", "brown fox jumps"}
    sig = minhash_signature(shingles, num_hashes=32)
    assert isinstance(sig, list), "MinHash signature must be a list."
    assert len(sig) == 32, "Signature length must match requested num_hashes."
    
    # Edge case: Empty shingle set
    empty_sig = minhash_signature(set(), num_hashes=16)
    assert len(empty_sig) == 16, "Edge case failed: MinHash length mismatched for empty shingles."
    assert all(val == 0 for val in empty_sig), "Edge case failed: Empty shingle signature must default to 0s."

def test_lsh_buckets():
    random.seed(42)
    sig = [random.randint(0, 1000) for _ in range(128)]
    buckets = lsh_buckets(sig, bands=16)
    assert isinstance(buckets, list), "Output must be a list."
    assert len(buckets) == 16, "Number of generated buckets must match bands count."
    assert len(buckets[0]) == 2, "Each bucket element must be a (band_id, bucket_hash) tuple."

def test_deduplicate():
    docs = [
        "Natural language processing is a subfield of linguistics and computer science.",
        "Natural language processing is a subfield of linguistics and computer science.",
        "Computer vision allows computers to derive meaningful information from digital images."
    ]
    deduped, removed = deduplicate(docs, threshold=0.8)
    assert isinstance(deduped, list), "Deduplicated docs must be returned as a list."
    assert len(deduped) == 2, "Duplicate document was not removed."
    assert removed == 1, "Removed duplicate count is incorrect."
    
    # Edge case: Single document input
    single_doc = ["Single document text."]
    d_out, r_out = deduplicate(single_doc)
    assert len(d_out) == 1 and r_out == 0, "Edge case failed: Single document should yield no duplicates."

def test_simple_tokenizer():
    tok = SimpleTokenizer()
    text = "hello world hello world"
    tok.train_bpe(text, num_merges=2)
    
    encoded = tok.encode("hello world")
    assert isinstance(encoded, list), "Encoded tokens must be a list of IDs."
    assert len(encoded) > 0, "Encoded sequence should not be empty."
    
    decoded = tok.decode(encoded)
    assert isinstance(decoded, str), "Decoded output must be a string."
    assert decoded == "hello world", "Decoded text does not match original."
    
    # Edge case: Out of vocabulary / raw unseen bytes
    unseen = tok.encode("xyz123")
    assert tok.decode(unseen) == "xyz123", "Edge case failed: Unseen characters were not decoded back accurately."

def test_tokenize_corpus():
    tok = SimpleTokenizer()
    tok.train_bpe("test document", num_merges=1)
    docs = ["test", "document"]
    tokens = tokenize_corpus(docs, tok)
    
    assert isinstance(tokens, list), "Corpus tokens must be a list."
    assert tokens.count(tok.eos_id) == 2, "Corpus token sequence must include one EOS token per document."

def test_pack_sequences():
    tokens = list(range(10))
    seqs, masks = pack_sequences(tokens, seq_length=4, pad_id=0)
    
    assert len(seqs) == 3, "Packed sequence count mismatch."
    assert len(seqs[0]) == 4, "Sequence block length mismatch."
    assert seqs[2] == [8, 9, 0, 0], "Padding incorrect on last sequence block."
    assert masks[2] == [1, 1, 0, 0], "Attention mask incorrect on padded sequence block."
    
    # Edge case: Empty token input
    e_seqs, e_masks = pack_sequences([], seq_length=4)
    assert len(e_seqs) == 0 and len(e_masks) == 0, "Edge case failed: Empty tokens should return empty lists."

def test_pre_training_data_loader():
    seqs = [[1, 2, 3, 4] for _ in range(10)]
    masks = [[1, 1, 1, 1] for _ in range(10)]
    loader = PreTrainingDataLoader(seqs, masks, batch_size=4, shuffle=False)
    
    assert len(loader) == 3, "DataLoader batch count mismatch."
    
    batches = list(loader)
    first_b_seqs, first_b_masks = batches[0]
    assert len(first_b_seqs) == 4, "Batch size mismatch."
    assert len(first_b_masks) == 4, "Attention mask batch size mismatch."
    
    # Edge case: Batch size greater than dataset size
    loader_large = PreTrainingDataLoader(seqs[:2], masks[:2], batch_size=8)
    large_batches = list(loader_large)
    assert len(large_batches) == 1, "Edge case failed: Oversized batch should return exactly 1 mini-batch."
    assert len(large_batches[0][0]) == 2, "Edge case failed: Single batch size should match total available samples."

def test_compute_statistics():
    docs = ["one two three", "four five"]
    token_ids = [1, 2, 3, 4, 5]
    sequences = [[1, 2, 3], [4, 5, 0]]
    
    stats = compute_statistics(docs, token_ids, sequences, tokenizer_vocab_size=256)
    assert isinstance(stats, dict), "Statistics output must be a dictionary."
    assert stats["total_documents"] == 2, "Stat mismatch: Total documents."
    assert stats["total_tokens"] == 5, "Stat mismatch: Total tokens."
    assert "compression_ratio" in stats, "Missing metric: compression_ratio."
    assert 0.0 <= stats["sequence_utilization"] <= 1.0, "Sequence utilization ratio out of valid bounds [0, 1]."


if __name__ == "__main__":
    run_pipeline()