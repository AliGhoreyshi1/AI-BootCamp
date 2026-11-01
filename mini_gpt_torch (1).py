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

import torch
import torch.nn as nn


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
        raise NotImplementedError("Implement this method")


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
        raise NotImplementedError("Implement this method")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Causal Multi-Head Attention module.

        Args:
            embed_dim (int): Total dimensionality of input and output features.
            num_heads (int): Number of parallel attention heads. Must divide embed_dim.

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
        raise NotImplementedError("Implement this method")


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
        raise NotImplementedError("Implement this method")


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        """
        Pre-LayerNorm Transformer block.

        Args:
            embed_dim (int): Embedding feature dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Intermediate feed-forward layer dimension.

        Attributes:
            ln1 (LayerNorm): Normalization applied before attention.
            attn (MultiHeadAttention): Causal self-attention sub-layer.
            ln2 (LayerNorm): Normalization applied before the feed-forward network.
            ffn (FeedForward): Feed-forward sub-layer.
        """
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

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
        raise NotImplementedError("Implement this method")


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
    raise NotImplementedError("Implement this function")


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        """
        Full MiniGPT causal language model.

        Args:
            vocab_size (int): Size of the vocabulary. Defaults to 50257.
            embed_dim (int): Hidden dimension size. Defaults to 768.
            num_heads (int): Number of attention heads. Defaults to 12.
            num_layers (int): Number of stacked Transformer blocks. Defaults to 12.
            max_seq_len (int): Maximum sequence context length. Defaults to 1024.
            ff_dim (int): Expansion dimension for the feed-forward network. Defaults to 3072.

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
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

    def forward(self, token_ids):
        """
        Forward pass converting token sequences into next-token logits.

        Args:
            token_ids (torch.Tensor): Batch of token index sequences, dtype torch.long.
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Unnormalized scores over the vocabulary (logits), produced by
                weight tying with the token embedding matrix. A causal mask built from
                `seq_len` must prevent every position from attending to later positions.
                Shape: (batch_size, seq_len, vocab_size)
        """
        raise NotImplementedError("Implement this method")

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
        raise NotImplementedError("Implement this method")


def cross_entropy_loss(logits, targets):
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

    Returns:
        torch.Tensor: Scalar (0-dimensional) loss tensor, averaged over all
            batch_size * seq_len positions.
            Shape: ()
    """
    raise NotImplementedError("Implement this function")


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
    raise NotImplementedError("Implement this function")


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
    raise NotImplementedError("Implement this function")


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
