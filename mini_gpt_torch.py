"""
Mini GPT — PyTorch implementation exercise.

ALLOWED PyTorch APIs:
    torch tensor operations, torch.nn.Parameter, torch.nn.Linear, torch.nn.Embedding,
    torch.nn.Module, torch.autograd, torch.optim.

BANNED PyTorch APIs (you must implement these yourself):
    torch.nn.LayerNorm, torch.nn.functional.layer_norm,
    torch.nn.MultiheadAttention, torch.nn.functional.scaled_dot_product_attention,
    torch.nn.Transformer / nn.TransformerEncoderLayer / nn.TransformerDecoderLayer,
    torch.nn.functional.softmax, torch.softmax, Tensor.softmax,
    torch.nn.functional.cross_entropy, torch.nn.functional.log_softmax,
    torch.nn.CrossEntropyLoss.

Gradients are handled entirely by autograd — you never write a backward pass. Every
forward pass you write must therefore stay differentiable: build outputs from tensor
operations on the inputs, and never call .detach(), .item(), .numpy(), or wrap
anything in torch.no_grad() except where a docstring explicitly says so.

All tensors are float32 unless stated otherwise, except `token_ids`, which is
torch.long. Do not hardcode dtypes or devices inside forward passes: derive them from
the incoming tensors, so the same code runs unchanged in float64.
"""
import math
import os
import torch
import torch.nn as nn
from data_pipeline import (
    PackedBlockLoader,
    clean_text,
    build_training_data,
    comments_to_documents,
    compute_statistics,
    split_packed,
)

# Label value written by data_pipeline.pack_documents() for padding / document boundaries.
IGNORE_INDEX = -100

def _softmax_last(x):
    """
    Numerically stable softmax over the last dimension, written with tensor operations only.

    Args:
        x (torch.Tensor): Scores of any shape.

    Returns:
        torch.Tensor: Probabilities with the same shape; the last dimension sums to 1.
    """
    shifted = x - x.amax(dim=-1, keepdim=True)
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=-1, keepdim=True)


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        """
        Token and positional embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of embedding vectors.
            max_seq_len (int): Maximum sequence length supported by positional embeddings.

        Attributes:
            token_embed (nn.Embedding): Token embedding table. Weight shape: (vocab_size, embed_dim)
            pos_embed (nn.Embedding): Positional embedding table. Weight shape: (max_seq_len, embed_dim)
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        """
        Computes combined token and positional embeddings for input token sequences.

        Args:
            token_ids (torch.Tensor): Token indices, dtype torch.long.
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Sum of token embeddings and the positional embeddings for
                positions 0..seq_len-1, broadcast across the batch.
                Shape: (batch_size, seq_len, embed_dim)
        """
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.token_embed(token_ids) + self.pos_embed(positions)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """
        Layer Normalization across the feature dimension.

        Args:
            dim (int): Feature/embedding dimension to normalize.
            eps (float): Epsilon added to the variance for numerical stability.

        Attributes:
            gamma (nn.Parameter): Learnable scale, initialized to ones. Shape: (dim,)
            beta (nn.Parameter): Learnable shift, initialized to zeros. Shape: (dim,)
            eps (float): Stored epsilon value.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """
        Normalizes the last dimension of the input tensor and applies scale and shift.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (..., dim)

        Returns:
            torch.Tensor: Layer-normalized tensor with the same shape as the input.
                The mean and the biased variance are computed over the last axis only.
                Shape: (..., dim)
        """
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        Causal Multi-Head Attention module.

        Args:
            embed_dim (int): Total dimensionality of input and output features.
            num_heads (int): Number of parallel attention heads. Must divide embed_dim.
            dropout (float): Dropout probability applied to the attention weights. Defaults to 0.0.

        Attributes:
            num_heads (int): Number of attention heads.
            head_dim (int): embed_dim // num_heads.
            W_q (nn.Linear): Query projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_k (nn.Linear): Key projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_v (nn.Linear): Value projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_out (nn.Linear): Output projection, no bias. Weight shape: (embed_dim, embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)
        for layer in (self.W_q, self.W_k, self.W_v, self.W_out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        """
        Multi-head projection, scaled dot-product attention with optional additive
        masking, and output projection. The softmax must be implemented by hand.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Additive attention mask, added to the
                attention scores before the softmax. Allowed positions hold 0.0, masked
                positions hold a large negative value (see `causal_mask`).
                Shape: (seq_len, seq_len) or broadcastable to
                (batch_size, num_heads, seq_len, seq_len). Defaults to None (no masking).

        Returns:
            torch.Tensor: Attention output after the heads are recombined and passed
                through W_out.
                Shape: (batch_size, seq_len, embed_dim)
        """
        batch, seq_len, embed_dim = x.shape
        q = self.W_q(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = self.attn_dropout(_softmax_last(scores))
        out = (weights @ v).transpose(1, 2).reshape(batch, seq_len, embed_dim)
        return self.W_out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        """
        Position-wise Feed-Forward Network (MLP).

        Args:
            embed_dim (int): Model embedding feature dimension.
            ff_dim (int): Hidden feature dimension of the expansion layer.

        Attributes:
            fc1 (nn.Linear): Expansion layer. Weight shape: (ff_dim, embed_dim), bias: (ff_dim,)
            fc2 (nn.Linear): Contraction layer. Weight shape: (embed_dim, ff_dim), bias: (embed_dim,)
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        for layer in (self.fc1, self.fc2):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Two-layer feed-forward transformation with a ReLU activation in between.

        Args:
            x (torch.Tensor): Input hidden states.
                Shape: (..., embed_dim)

        Returns:
            torch.Tensor: Transformed features, projected up to ff_dim and back down.
                Shape: (..., embed_dim)
        """
        hidden = self.fc1(x).clamp(min=0.0)
        return self.fc2(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        """
        Pre-LayerNorm Transformer block.

        Args:
            embed_dim (int): Embedding feature dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Intermediate feed-forward layer dimension.
            dropout (float): Dropout probability for attention weights and for the output of
                each sub-layer (before the residual addition). Defaults to 0.0.

        Attributes:
            ln1 (LayerNorm): Normalization applied before attention.
            attn (MultiHeadAttention): Causal self-attention sub-layer.
            ln2 (LayerNorm): Normalization applied before the feed-forward network.
            ffn (FeedForward): Feed-forward sub-layer.
        """
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Passes the input through the attention and feed-forward sub-layers, each with
        pre-normalization and a residual connection.

        Args:
            x (torch.Tensor): Input representation.
                Shape: (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Additive causal mask forwarded to attention.
                Shape: (seq_len, seq_len) or broadcastable. Defaults to None.

        Returns:
            torch.Tensor: Output representation with the same shape as the input.
                Shape: (batch_size, seq_len, embed_dim)
        """
        x = x + self.resid_dropout(self.attn(self.ln1(x), mask))
        x = x + self.resid_dropout(self.ffn(self.ln2(x)))
        return x


def causal_mask(seq_len, dtype=torch.float32, device=None):
    """
    Builds the additive causal (autoregressive) attention mask.

    Args:
        seq_len (int): Sequence length.
        dtype (torch.dtype): Data type of the returned mask. Defaults to torch.float32.
        device (torch.device, optional): Device of the returned mask. Defaults to None (CPU).

    Returns:
        torch.Tensor: Square additive mask whose entry [i, j] is 0.0 when position i is
            allowed to attend to position j (j <= i) and a large negative value
            otherwise. Use torch.finfo(dtype).min rather than a hardcoded constant so
            the mask stays valid in float64.
            Shape: (seq_len, seq_len)
    """
    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=1)
    return upper * torch.finfo(dtype).min


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072, dropout=0.0):
        """
        Full MiniGPT causal language model.

        Args:
            vocab_size (int): Size of the vocabulary. Defaults to 50257.
            embed_dim (int): Hidden dimension size. Defaults to 768.
            num_heads (int): Number of attention heads. Defaults to 12.
            num_layers (int): Number of stacked Transformer blocks. Defaults to 12.
            max_seq_len (int): Maximum sequence context length. Defaults to 1024.
            ff_dim (int): Expansion dimension for the feed-forward network. Defaults to 3072.
            dropout (float): Dropout probability on the embeddings, attention weights and
                sub-layer outputs. Use 0.0 for evaluation-only models. Defaults to 0.0.

        Attributes:
            embedding (Embedding): Joint token and positional embedding layer.
            blocks (nn.ModuleList): Stack of `num_layers` TransformerBlock modules.
            ln_f (LayerNorm): Final normalization applied before the output projection.
            vocab_size (int): Stored vocabulary size.
            embed_dim (int): Stored embedding dimension.
            max_seq_len (int): Stored maximum sequence length.
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.ff_dim = ff_dim

    def forward(self, token_ids, doc_ids=None):
        """
        Forward pass converting token sequences into next-token logits.

        Args:
            token_ids (torch.Tensor): Batch of token index sequences, dtype torch.long.
                Shape: (batch_size, seq_len)
            doc_ids (torch.Tensor, optional): Document id of every token, dtype torch.long,
                as produced by data_pipeline.pack_documents(). When given, a token may only
                attend to earlier tokens of its own document (block-diagonal causal mask),
                so packed documents do not see each other. Defaults to None (plain causal).
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Unnormalized scores over the vocabulary (logits), produced by
                weight tying with the token embedding matrix. A causal mask built from
                `seq_len` must prevent every position from attending to later positions.
                Shape: (batch_size, seq_len, vocab_size)
        """
        x = self.embed_dropout(self.embedding(token_ids))
        mask = causal_mask(token_ids.shape[1], dtype=x.dtype, device=x.device)
        if doc_ids is not None:
            neg = torch.finfo(x.dtype).min
            other_doc = (doc_ids.unsqueeze(2) != doc_ids.unsqueeze(1)).to(x.dtype)  # (B, T, T)
            # causal + different-document can add up to -inf, so clamp back to the finite minimum.
            mask = (mask + other_doc.unsqueeze(1) * neg).clamp(min=neg)  # (B, 1, T, T)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return x @ self.embedding.token_embed.weight.transpose(0, 1)

    def count_parameters(self):
        """
        Computes the total number of trainable parameters from the architecture itself.

        Count the token and positional embedding tables, the four attention projections,
        both feed-forward weights and biases, both LayerNorm gains and shifts of every
        block, and the final LayerNorm parameters — deriving each size from the
        hyper-parameters. Do NOT use self.parameters(); the test compares your result
        against it.

        Returns:
            int: Grand total parameter count across all components.
        """
        V, D, L, F, n = self.vocab_size, self.embed_dim, self.max_seq_len, self.ff_dim, self.num_layers
        embeddings = V * D + L * D
        attention = 4 * D * D
        feed_forward = (D * F + F) + (F * D + D)
        layer_norms = 2 * (2 * D)
        per_block = attention + feed_forward + layer_norms
        final_ln = 2 * D
        return embeddings + n * per_block + final_ln


def cross_entropy_loss(logits, targets, ignore_index=None):
    """
    Computes the average cross-entropy loss over a batch of sequences.

    Must be implemented with a numerically stable log-softmax written by hand, and must
    stay differentiable: the returned tensor is what `.backward()` is called on during
    training, so do not detach it or convert it to a Python float.

    Args:
        logits (torch.Tensor): Model output logits before softmax.
            Shape: (batch_size, seq_len, vocab_size)
        targets (torch.Tensor): Ground-truth target token indices, dtype torch.long.
            Shape: (batch_size, seq_len)
        ignore_index (int, optional): Target value that is excluded from the loss (for
            example -100 for padding in packed batches). Defaults to None (no position
            is ignored).

    Returns:
        torch.Tensor: Scalar (0-dimensional) loss tensor, averaged over all
            batch_size * seq_len positions (or over the non-ignored positions only when
            ignore_index is given).
            Shape: ()
    """
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    shifted = flat_logits - flat_logits.amax(dim=-1, keepdim=True)
    log_partition = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_probs = shifted - log_partition
    if ignore_index is None:
        nll = -log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        return nll.mean()
    keep = flat_targets != ignore_index
    safe_targets = torch.where(keep, flat_targets, torch.zeros_like(flat_targets))  # valid index for gather
    nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    keep = keep.to(nll.dtype)
    return (nll * keep).sum() / keep.sum().clamp(min=1.0)


def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    """
    Autoregressively generates new tokens from a prompt using temperature sampling.

    Runs without gradient tracking. Sampling must use torch.multinomial, so that seeding
    with torch.manual_seed makes the output reproducible.

    Args:
        model (MiniGPT): The language model instance.
        prompt_tokens (list[int]): Initial prompt token IDs.
        max_new_tokens (int): Number of new tokens to generate. Defaults to 100.
        temperature (float): Divisor applied to the logits before the softmax. Lower
            values sharpen the distribution, higher values flatten it. Defaults to 0.8.

    Returns:
        list[int]: The prompt followed by the generated tokens, of total length
            len(prompt_tokens) + max_new_tokens. The context fed to the model at each
            step must be truncated to the model's maximum sequence length.
    """
    device = next(model.parameters()).device
    model.eval()
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[:, -model.max_seq_len:]
            logits = model(context)[:, -1, :] / temperature
            probs = _softmax_last(logits)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokens[0].tolist()


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=1500, lr=3e-4, batch_size=4):
    """
    Runs an end-to-end training loop for MiniGPT on raw text.

    The text is encoded as UTF-8 bytes, so the vocabulary is the 256 possible byte
    values. Each step samples `batch_size` random windows of length seq_len + 1, uses
    the first seq_len bytes of each window as input and the last seq_len bytes as
    targets, computes the loss, backpropagates with autograd and updates every
    parameter with a torch.optim.AdamW optimizer. Print the loss every 20 steps so
    training is visible.

    Sanity check: with the defaults, the loss starts near ln(256) = 5.55 and should
    fall below 0.5 within roughly 1500 steps (about a minute on a CPU). If it plateaus
    above 3.0, something in your forward pass or loss is wrong. Note that the optimizer
    choice is part of the specification: plain SGD at this learning rate barely moves
    the loss at all.

    Args:
        text (str): Raw input text corpus used for training.
        vocab_size (int): Size of the byte vocabulary. Defaults to 256.
        embed_dim (int): Model embedding dimensionality. Defaults to 128.
        num_heads (int): Attention head count. Defaults to 4.
        num_layers (int): Transformer depth. Defaults to 4.
        seq_len (int): Training context length window, also used as the model's
            maximum sequence length. Defaults to 64.
        num_steps (int): Total gradient update steps. Defaults to 1500.
        lr (float): AdamW learning rate. Defaults to 3e-4.
        batch_size (int): Number of sequences sampled per step. Defaults to 4.

    Returns:
        MiniGPT: The trained model instance, left in eval mode.
    """
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    if len(data) < seq_len + 1:
        raise ValueError("text is shorter than seq_len + 1 bytes")
    device = torch.device(device = ("cuda" if torch.cuda.is_available() else "cpu"))
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long, device=device)
    model = MiniGPT(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
                    num_layers=num_layers, max_seq_len=seq_len, ff_dim=4 * embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    max_start = len(data) - seq_len - 1
    model.train()
    for step in range(1, num_steps + 1):
        starts = torch.randint(0, max_start + 1, (batch_size,)).tolist()
        windows = torch.stack([data[s:s + seq_len + 1] for s in starts])
        inputs = windows[:, :seq_len]
        targets = windows[:, 1:]
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.4f}")
    model.eval()
    return model


def _batch_to_tensors(batch, device):
    """Convert a dict of python lists (from PackedBlockLoader) into long tensors on `device`."""
    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in batch.items()}


def evaluate_loss(model, packed, batch_size=16):
    """
    Mean next-token loss of `model` over packed blocks, without gradient tracking.

    Args:
        model (MiniGPT): Model to evaluate (its train/eval mode is restored afterwards).
        packed (dict): Blocks with keys input_ids, labels, doc_ids, attention_mask.
        batch_size (int): Blocks per evaluation batch.

    Returns:
        float: Loss averaged over all supervised (non-ignored) positions.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in PackedBlockLoader(packed, batch_size, shuffle=False):
            t = _batch_to_tensors(batch, device)
            logits = model(t["input_ids"], doc_ids=t["doc_ids"])
            n = int((t["labels"] != IGNORE_INDEX).sum())
            total += cross_entropy_loss(logits, t["labels"], ignore_index=IGNORE_INDEX).item() * n
            count += n
    model.train(was_training)
    return total / max(count, 1)


def train_mini_gpt_with_pipeline(csv_path, num_documents=None, num_merges=300, seq_length=128,
                                 embed_dim=128, num_heads=4, num_layers=4, num_steps=3000,
                                 lr=3e-4, batch_size=8, val_fraction=0.1, eval_every=100, seed=0,
                                 dropout=0.1, weight_decay=0.1, patience=5,
                                 checkpoint_path="best_mini_gpt.pt"):
    """
    Trains MiniGPT on the doctor-comments corpus, with all data preparation done by data_pipeline.

    Flow: comments_to_documents (group short comments into documents) -> build_training_data
    (clean with hazm, language/quality filter, MinHash dedup, PII removal, BPE training,
    tokenization, packing with doc ids and next-token labels) -> split_packed (train / validation
    blocks) -> PackedBlockLoader (shuffled batches) -> AdamW training with dropout, a
    document-aware attention mask and a loss that ignores padding and document boundaries.

    Early stopping: validation loss is measured every `eval_every` steps. The weights with the
    lowest validation loss are kept in memory (and saved to `checkpoint_path`). Training stops
    after `patience` evaluations without improvement, and the best weights are restored.

    Args:
        csv_path (str): Path of comments_raw_canonical.csv.
        num_documents (Optional[int]): Number of grouped documents to use. None uses all of them.
        num_merges (int): BPE merges learned by the pipeline tokenizer.
        seq_length (int): Block length, also the model's maximum sequence length.
        embed_dim (int): Model embedding size.
        num_heads (int): Attention heads.
        num_layers (int): Transformer blocks.
        num_steps (int): Maximum number of optimizer steps.
        lr (float): AdamW learning rate.
        batch_size (int): Blocks per step.
        val_fraction (float): Fraction of blocks held out to measure validation loss.
        eval_every (int): Measure validation loss every this many steps.
        seed (int): Seed for document sampling, batch shuffling and weight init.
        dropout (float): Dropout probability inside the model. Defaults to 0.1.
        weight_decay (float): AdamW weight decay for weight matrices (biases and LayerNorm
            parameters are not decayed). Defaults to 0.1.
        patience (int): Number of validation checks without improvement before stopping.
        checkpoint_path (Optional[str]): Where the best weights are saved. None disables saving.

    Returns:
        Tuple[MiniGPT, SimpleTokenizer, List[str]]: Best model (eval mode), the tokenizer
            created by the pipeline and the final documents that were used for training.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    documents = comments_to_documents(csv_path, max_documents=num_documents, seed=seed)
    tokenizer, packed, final_docs = build_training_data(
        documents, num_merges=num_merges, seq_length=seq_length)
    train_blocks, val_blocks = split_packed(packed, val_fraction)

    token_ids = [t for seq, m in zip(packed["input_ids"], packed["attention_mask"])
                 for t, keep in zip(seq, m) if keep]
    stats = compute_statistics(final_docs, token_ids, packed["input_ids"],
                               tokenizer.vocab_size(), pad_id=tokenizer.pad_id)
    print(f"documents: {stats['total_documents']} | tokens: {stats['total_tokens']:,} | "
          f"vocab: {tokenizer.vocab_size()} | blocks: train {len(train_blocks['input_ids'])} "
          f"/ val {len(val_blocks['input_ids'])} (length {seq_length})")
    print(f"bytes per token: {stats['compression_ratio']:.2f} | "
          f"initial loss should be near ln(vocab) = {math.log(tokenizer.vocab_size()):.2f}")

    model = MiniGPT(vocab_size=tokenizer.vocab_size(), embed_dim=embed_dim, num_heads=num_heads,
                    num_layers=num_layers, max_seq_len=seq_length, ff_dim=4 * embed_dim,
                    dropout=dropout).to(device)

    # Weight decay only for matrices (dim >= 2); biases and LayerNorm parameters are excluded.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr)
    loader = PackedBlockLoader(train_blocks, batch_size, shuffle=True, seed=seed)

    best_val, best_step, best_state, bad_evals = float("inf"), 0, None, 0
    model.train()
    step, epoch, stop = 0, 0, False
    while step < num_steps and not stop:
        loader.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            step += 1
            t = _batch_to_tensors(batch, device)
            logits = model(t["input_ids"], doc_ids=t["doc_ids"])
            loss = cross_entropy_loss(logits, t["labels"], ignore_index=IGNORE_INDEX)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 1 or step % 20 == 0:
                print(f"step {step:5d} | epoch {epoch:3d} | train loss {loss.item():.4f}")
            if step % eval_every == 0 or step == num_steps:
                val = evaluate_loss(model, val_blocks)
                print(f"step {step:5d} | validation loss {val:.4f}")
                if val < best_val - 1e-4:
                    best_val, best_step, bad_evals = val, step, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    if checkpoint_path:
                        torch.save(best_state, checkpoint_path)
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        print(f"early stopping at step {step}: no validation improvement "
                              f"in {patience} checks")
                        stop = True
                        break
            if step >= num_steps:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best validation loss {best_val:.4f} at step {best_step}"
          + (f" (saved to {checkpoint_path})" if checkpoint_path else ""))
    model.eval()
    return model, tokenizer, final_docs


def complete_text(model, tokenizer, prompt, max_new_tokens=60, temperature=0.7):
    """
    Encodes `prompt` with the pipeline tokenizer, samples a continuation with generate(),
    cuts it at the first end-of-text token and decodes it back to a string.

    Args:
        model (MiniGPT): Trained model.
        tokenizer (SimpleTokenizer): Tokenizer from data_pipeline.build_training_data.
        prompt (str): Text to continue.
        max_new_tokens (int): Number of tokens to sample.
        temperature (float): Sampling temperature.

    Returns:
        str: Prompt followed by the generated continuation.
    """
    # Normalize the prompt the same way as the training text (hazm via clean_text).
    prompt_ids = tokenizer.encode(clean_text(prompt))
    ids = generate(model, prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    new_ids = ids[len(prompt_ids):]
    if tokenizer.eos_id in new_ids:
        new_ids = new_ids[:new_ids.index(tokenizer.eos_id)]
    return tokenizer.decode(prompt_ids + new_ids)


def parameter_breakdown():
    """
    Prints parameter counts for the standard GPT-2 configuration sizes.

    Returns:
        None
    """
    configs = [
        ("GPT-2 Small", 50257, 768, 12, 12, 1024, 3072),
        ("GPT-2 Medium", 50257, 1024, 16, 24, 1024, 4096),
        ("GPT-2 Large", 50257, 1280, 20, 36, 1024, 5120),
        ("GPT-2 XL", 50257, 1600, 25, 48, 1024, 6400),
    ]
    print("GPT-2 Family Parameter Counts")
    print("=" * 65)
    print(f"{'Model':<16} {'Layers':>6} {'Heads':>6} {'Dims':>6} {'Params':>14}")
    print("-" * 65)
    for name, vocab, dim, heads, layers, seq_len, ff in configs:
        token_emb = vocab * dim
        pos_emb = seq_len * dim
        per_block_attn = 4 * dim * dim
        per_block_ff = 2 * dim * ff + dim + ff
        per_block_ln = 4 * dim
        per_block = per_block_attn + per_block_ff + per_block_ln
        final_ln = 2 * dim
        total = token_emb + pos_emb + layers * per_block + final_ln
        print(f"{name:<16} {layers:>6} {heads:>6} {dim:>6} {total:>14,}")
    print()


def memory_estimate():
    """
    Prints theoretical FP16 inference memory consumption for several modern models.

    Returns:
        None
    """
    print("Memory Requirements for Inference (FP16)")
    print("=" * 65)
    models = [
        ("GPT-2 Small (124M)", 124e6, 12, 12, 64, 1024),
        ("Llama 3 8B", 8e9, 32, 32, 128, 8192),
        ("Llama 3 70B", 70e9, 80, 64, 128, 8192),
        ("Llama 3 405B", 405e9, 126, 128, 128, 131072),
    ]
    print(f"{'Model':<24} {'Weights':>10} {'KV Cache':>12} {'Total':>10}")
    print("-" * 65)
    for name, params, layers, heads, head_dim, max_seq in models:
        weight_bytes = params * 2
        kv_per_token = 2 * layers * heads * head_dim * 2
        kv_full = kv_per_token * max_seq
        total = weight_bytes + kv_full

        def fmt(b):
            if b >= 1e9:
                return f"{b / 1e9:.1f} GB"
            return f"{b / 1e6:.0f} MB"

        print(f"{name:<24} {fmt(weight_bytes):>10} {fmt(kv_full):>12} {fmt(total):>10}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniGPT demos")
    parser.add_argument("--csv", default="comments_raw_canonical.csv",
                        help="doctor comments CSV used for the data_pipeline training demo")
    parser.add_argument("--docs", type=int, default=None, help="documents sampled from the CSV (default: all)")
    parser.add_argument("--merges", type=int, default=300, help="BPE merges for the pipeline tokenizer")
    parser.add_argument("--steps", type=int, default=3000, help="maximum training steps for the pipeline demo")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--patience", type=int, default=5, help="validation checks without improvement before stopping")
    parser.add_argument("--only-pipeline", action="store_true", help="skip the tiny built-in corpus demo")
    parser.add_argument("--no-pipeline", action="store_true", help="skip the doctor-comments demo")
    args = parser.parse_args()

    if not args.only_pipeline:
        torch.manual_seed(42)
        parameter_breakdown()
        memory_estimate()
        corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""
        print("Training Mini GPT")
        print("=" * 65)
        model = train_mini_gpt(corpus, num_steps=1500)
        prompt = list("The transformer".encode("utf-8"))
        print(f"\nPrompt: 'The transformer'")
        print("Generating...")
        output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.6)
        generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
        print(f"Generated: {generated_text}")

    if not args.no_pipeline:
        if not os.path.exists(args.csv):
            print(f"\n{args.csv} not found - skipping the data_pipeline demo (use --csv PATH).")
        else:
            print("\nTraining Mini GPT on doctor comments via data_pipeline")
            print("=" * 65)
            model, tokenizer, docs = train_mini_gpt_with_pipeline(
                args.csv, num_documents=args.docs, num_merges=args.merges, num_steps=args.steps,
                dropout=args.dropout, patience=args.patience)
            seen_prompt = " ".join(docs[0].split()[:6])
            print(f"\nPrompt taken from a training document: {seen_prompt!r}")
            print("Model :", complete_text(model, tokenizer, seen_prompt, max_new_tokens=60, temperature=0.5))
            print("Actual:", " ".join(docs[0].split()[:25]))
            for prompt in ("دکتر", "بسیار", "از"):
                print(f"\nPrompt: {prompt!r}")
                print(complete_text(model, tokenizer, prompt, max_new_tokens=60, temperature=0.7))
