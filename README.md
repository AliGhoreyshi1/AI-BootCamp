# AI-BootCamp

A hands-on implementation of core language-modeling components, built with Python and PyTorch. The project explores text tokenization, Persian text preprocessing, training-data preparation, and a compact GPT-style Transformer trained with next-token prediction.

> **Project status:** Educational / experimental. This repository is intended for learning and experimentation; it is not a production-ready language model or a medical application.

## Overview

AI-BootCamp is organized around three connected areas:

1. **Tokenization** – character-level and Byte Pair Encoding (BPE) tokenizers, including a byte-level tokenizer with pre-tokenization and special-token support.
2. **Dataset preparation** – cleaning and normalizing text, filtering low-quality documents, detecting dominant scripts/languages, removing personally identifiable information (PII), detecting near-duplicates, and packing token sequences for language-model training.
3. **MiniGPT** – a small GPT-like autoregressive Transformer implemented in PyTorch, with causal self-attention, training and validation loops, text generation, and checkpointing.

The included CSV contains Persian-language comments associated with doctors. The training pipeline groups short comments by doctor to create longer documents before preparing them for pre-training.

## Features

### Tokenizers

- `CharTokenizer`: converts characters to integer code points and back.
- `BPETokenizer`: learns byte-pair merges from a corpus and supports encoding and decoding.
- `ProductionTokenizer`: extends BPE with Unicode NFKC normalization, GPT-2-style regex pre-tokenization, byte-level encoding, and configurable special tokens.
- Tokenizer demonstrations, round-trip checks, vocabulary statistics, and optional comparison with `tiktoken`.

### Data pipeline

- Persian normalization using Hazm, including normalization of Arabic/Persian character variants and spacing.
- Removal of HTML tags and URLs, followed by whitespace cleanup.
- Quality filtering based on document length, uppercase-word ratio, and special-character ratio.
- Script/language heuristics for filtering documents by dominant language.
- Near-duplicate detection using word shingles, MinHash signatures, Locality-Sensitive Hashing (LSH), and Jaccard similarity.
- PII-pattern removal and repetitive-text filtering.
- Grouping short doctor comments into longer documents.
- BPE training, tokenization, sequence packing, document-aware labels and IDs, train/validation splitting, and batch loading.
- Dataset statistics such as token counts, vocabulary utilization, compression ratio, and sequence utilization.

### MiniGPT model

- Custom token and positional embeddings.
- Custom Layer Normalization implementation.
- Causal multi-head self-attention and feed-forward blocks.
- Residual Transformer blocks and autoregressive next-token prediction.
- Padding/document-boundary-aware loss masking and document-aware attention.
- AdamW optimization, validation-loss tracking, early stopping, and best-checkpoint saving.
- Temperature-based autoregressive text generation.
- Automatic CUDA selection when a CUDA-enabled PyTorch installation and compatible GPU are available; otherwise, CPU is used.

## Repository structure

```text
AI-BootCamp/
├── comments_raw_canonical.csv  # Doctor-comment dataset used by the training pipeline
├── data_pipeline.py            # Text cleaning, filtering, deduplication and data preparation
├── mini_gpt_torch.py           # PyTorch MiniGPT model, training and generation
├── tokenizer1.py               # Character tokenizer and foundational BPE implementation
├── tokenizer2.py               # Byte-level production-style BPE tokenizer
└── README.md
```

## Requirements

- Python 3.10 or newer is recommended.
- PyTorch.
- Hazm.
- `regex` (recommended; `tokenizer2.py` includes a standard-library `re` fallback).
- `tiktoken` is optional and is only needed for the tokenizer comparison demonstrations.

Install the core dependencies:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
python -m pip install torch hazm regex
```

For GPU training, install the PyTorch build that matches your CUDA environment. Follow the official selector at <https://pytorch.org/get-started/locally/> rather than assuming that a generic PyPI installation will enable CUDA.

Optional dependency for tokenizer comparison demos:

```bash
python -m pip install tiktoken
```

## Getting started

Clone the repository and move into its directory:

```bash
git clone <your-repository-url>
cd AI-BootCamp
```

Make sure `comments_raw_canonical.csv` is in the project root, alongside `mini_gpt_torch.py`.

### Run MiniGPT

Run both the built-in small-corpus demonstration and the doctor-comments pipeline:

```bash
python mini_gpt_torch.py
```

Run only the doctor-comments training pipeline:

```bash
python mini_gpt_torch.py --only-pipeline
```

Use a subset of documents and customize training:

```bash
python mini_gpt_torch.py --only-pipeline --docs 5000 --merges 300 --steps 3000
```

Available command-line options:

| Option | Description | Default |
|---|---|---:|
| `--csv` | Path to the input CSV file | `comments_raw_canonical.csv` |
| `--docs` | Maximum number of grouped documents to sample; omit to use all | All |
| `--merges` | Number of BPE merge operations | `300` |
| `--steps` | Maximum optimizer steps | `3000` |
| `--dropout` | Model dropout probability | `0.1` |
| `--patience` | Number of validation checks without improvement before early stopping | `5` |
| `--only-pipeline` | Skip the built-in sample-corpus demo | Off |
| `--no-pipeline` | Skip the doctor-comments dataset demo | Off |

To use a CSV stored elsewhere:

```bash
python mini_gpt_torch.py --only-pipeline --csv path/to/comments_raw_canonical.csv
```

The training function saves the best model weights to `best_mini_gpt.pt` by default. The checkpoint contains the model state dictionary; keep the tokenizer configuration/merge rules alongside it if you plan to reproduce text generation later.

### Run tokenizer demonstrations

```bash
python tokenizer1.py
python tokenizer2.py
```

These scripts run their built-in tests and demonstrations. The `tokenizer1.py` comparison with `tiktoken` is skipped when that optional package is not installed.

### Run the standalone data-pipeline demonstration

```bash
python data_pipeline.py
```

This runs the pipeline's built-in sample-corpus demonstration and its assertions. The doctor-comments CSV training workflow is launched through `mini_gpt_torch.py` as shown above.

## Dataset format

The provided `comments_raw_canonical.csv` uses the following columns:

| Column | Description |
|---|---|
| `doctor_id` | Identifier used to group comments from the same doctor |
| `source_file` | Name of the source file |
| `comment_index_in_file` | Original comment position in the source file |
| `specialty` | Doctor specialty |
| `text` | Comment text used by the language-modeling pipeline |
| `rate` | Rating value included in the source dataset |
| `label` | Label included in the source dataset |
| `date` | Comment date |

The current document-building function reads `text` and `doctor_id`. It normalizes comments, removes exact duplicates within each doctor, joins comments until a document reaches the configured minimum word count, shuffles the resulting documents with a seed, and optionally samples a maximum number of documents.

## High-level training workflow

```text
CSV comments
    │
    ▼
Group comments by doctor and build longer documents
    │
    ▼
Normalize and clean text
    │
    ▼
Filter low-quality / unsupported / repetitive content
    │
    ▼
Remove near-duplicates and PII patterns
    │
    ▼
Train byte-level BPE tokenizer and encode documents
    │
    ▼
Pack token IDs with document IDs, labels and masks
    │
    ▼
Create training and validation blocks
    │
    ▼
Train MiniGPT with causal attention and next-token loss
    │
    ▼
Save best checkpoint and generate sample text
```

## Notes and limitations

- This is a compact educational implementation. Training quality depends heavily on dataset size and quality, tokenizer settings, model capacity, and training duration.
- The language detector is a Unicode-script heuristic, not a full language-identification model. In particular, it does not distinguish English from other Latin-script languages.
- PII removal is pattern-based and cannot guarantee that all identifying information is removed. Review data handling carefully before sharing or using the dataset.
- The included data appears to contain user comments about healthcare providers. Confirm that you have the necessary rights and permissions to use or redistribute it. Do not publish private, identifying, or otherwise restricted data in a public repository.
- GPU acceleration requires a compatible NVIDIA driver, CUDA-enabled PyTorch build, and supported hardware. The code falls back to CPU if CUDA is unavailable.
- The training script's default run includes a small built-in demonstration before the CSV workflow. Use `--only-pipeline` to skip that first demonstration.

## Tests

The tokenizer scripts execute their built-in test functions when run directly:

```bash
python tokenizer1.py
python tokenizer2.py
```

The data-pipeline script also runs its assertions when executed:

```bash
python data_pipeline.py
```

## License

No license file was included with the supplied project files. Add a `LICENSE` file before distributing this repository, and ensure that the dataset's license and usage permissions are compatible with your intended distribution.

---

**AI-BootCamp** — learning how tokenizers, data pipelines, and Transformer language models fit together.
