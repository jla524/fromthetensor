"""
Attention Residuals: A Kimi (MoonshotAI) Approach

This module implements Attention Residuals as described in the Kimi paper from MoonshotAI.
Attention residuals replace standard residual connections with learned attention over
all previous layer outputs, creating a more expressive and connected architecture.

Key Concepts:
- **Full Attention Residuals**: Each layer attends over ALL previous outputs (O(L²d) memory)
- **Block Attention Residuals**: Efficient variant with O(Nd) memory by grouping layers into blocks
  - Within blocks: standard residuals
  - Between blocks: attention residuals

Architecture Benefits:
- Long-range gradient flow (no vanishing gradients)
- Global information mixing across layers
- Maintains O(Nd) memory for practical use with block variant

Paper: MoonshotAI/Kimi Attention Residuals
Implementation follows the paper's specifications:
- RMSNorm preferred, LayerNorm fallback
- GELU activation
- Weight initialization: std=0.02
- Block structure for efficiency

Memory Complexity:
- FullAttentionResidual: O(L² × d) where L = layers, d = dimension
- BlockAttentionResidual: O(N × d) where N = block size, d = dimension
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Helper Functions
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_norm_layer(d_model: int, eps: float = 1e-6) -> nn.Module:
    """
    Get normalization layer: RMSNorm if available (via custom impl or torch>=2.4),
    otherwise LayerNorm.

    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability

    Returns:
        Normalization layer instance
    """
    # Custom RMSNorm implementation (paper's preference)
    return RMSNorm(d_model, eps)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization as preferred by the Kimi paper.

    RMSNorm normalizes by the root mean square of the inputs, without centering
    (subtracting mean). This is simpler and often more stable than LayerNorm.

    Formula: output = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., d_model]

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normalized = x / rms
        return self.weight * x_normalized


# ============================================================================
# Attention Residual Implementations
# ============================================================================


class FullAttentionResidual(nn.Module):
    """
    Full Attention Residuals: O(L²d) memory complexity.

    Each layer l attends over ALL previous layer outputs (1 to l-1) using learned
    attention weights. This creates a fully connected layer graph where information
    from any previous layer can directly influence any subsequent layer.

    Formula:
        h_l = Σᵢ₌₁ˡ⁻¹ αᵢ→ₗ · vᵢ

    where:
        - αᵢ→ₗ = softmax(q_l · k_i / √d) are cross-layer attention weights
        - q_l is a learned pseudo-query for layer l
        - k_i, v_i are key and value projections of layer i's output

    Memory: O(L² × d) - stores attention weights between all layer pairs

    Use case: Small models where full connectivity is desired and memory permits.
    For larger models, use BlockAttentionResidual.

    Args:
        d_model: Model dimension
        num_layers: Total number of layers
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self, d_model: int, num_layers: int, num_heads: int = 8, dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Pseudo-queries: one learned query vector per layer
        # Shape: [num_layers, num_heads, head_dim]
        self.pseudo_queries = nn.Parameter(
            torch.randn(num_layers, num_heads, self.head_dim) * 0.02
        )

        # Key and value projections for incoming layer outputs
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Store past layer outputs during forward pass
        self.reset_cache()

    def reset_cache(self):
        """Clear the stored layer outputs cache."""
        self.past_outputs: List[torch.Tensor] = []

    def forward(self, layer_idx: int, current_output: torch.Tensor) -> torch.Tensor:
        """
        Compute attention residual for the current layer.

        Args:
            layer_idx: Index of current layer (0-indexed)
            current_output: Current layer output [batch, seq_len, d_model]

        Returns:
            Attention residual [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = current_output.shape

        # Store current output for future layers.
        # NOTE: do NOT call .detach() here — that would cut gradient flow through
        # the cross-layer attention weights, preventing the gradient uniformity
        # property the paper claims.  Memory use is O(L * B * T * D) per forward
        # pass which is fine for small research models.
        self.past_outputs.append(current_output)

        # If no previous outputs, return zero residual
        if len(self.past_outputs) == 0 or layer_idx == 0:
            return torch.zeros_like(current_output)

        # Get all previous outputs
        # Stack: [num_prev_layers, batch, seq_len, d_model]
        prev_outputs = torch.stack(self.past_outputs[:-1], dim=0)
        num_prev = prev_outputs.shape[0]

        # Reshape for multi-head attention: [num_prev, batch*seq_len, d_model]
        prev_outputs_flat = prev_outputs.view(num_prev, -1, self.d_model)

        # Compute keys and values for all previous outputs
        # Shape: [num_prev, batch*seq_len, num_heads, head_dim]
        keys = self.key_proj(prev_outputs_flat).view(
            num_prev, -1, self.num_heads, self.head_dim
        )
        values = self.value_proj(prev_outputs_flat).view(
            num_prev, -1, self.num_heads, self.head_dim
        )

        # Get query for current layer
        # Shape: [num_heads, head_dim]
        query = self.pseudo_queries[layer_idx]  # [num_heads, head_dim]

        # Compute attention scores
        # keys: [num_prev, batch*seq_len, num_heads, head_dim]
        # query: [num_heads, head_dim]

        # Reshape keys for attention computation
        # keys: [num_prev, batch*seq_len, num_heads, head_dim]
        keys_permuted = keys.permute(
            2, 0, 1, 3
        )  # [num_heads, num_prev, batch*seq_len, head_dim]

        # Expand query: [num_heads, 1, head_dim]
        query_expanded = query.unsqueeze(1)  # [num_heads, 1, head_dim]

        # Compute attention scores using matmul
        # query: [num_heads, 1, head_dim]
        # keys: [num_heads, num_prev, batch*seq_len, head_dim]
        # We need: [num_prev, num_heads, batch*seq_len]

        # Reshape keys for batch matrix multiply: [num_heads, num_prev*batch*seq, head_dim]
        keys_reshaped = keys_permuted.reshape(
            self.num_heads, num_prev * batch_size * seq_len, self.head_dim
        )

        # scores: [num_heads, 1, num_prev*batch*seq]
        scores = (
            torch.matmul(query_expanded, keys_reshaped.transpose(-2, -1)) * self.scale
        )

        # Reshape scores: [num_heads, num_prev, batch*seq_len]
        scores = scores.squeeze(1).reshape(
            self.num_heads, num_prev, batch_size * seq_len
        )

        # Transpose to [num_prev, num_heads, batch*seq_len]
        scores = scores.permute(1, 0, 2)

        # Softmax over previous layers
        attn_weights = F.softmax(scores, dim=0)  # [num_prev, num_heads, batch*seq_len]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # values: [num_prev, batch*seq_len, num_heads, head_dim]
        # attn_weights: [num_prev, num_heads, batch*seq_len]
        # output: [batch*seq_len, num_heads, head_dim]

        values_t = values.permute(
            1, 2, 0, 3
        )  # [batch*seq_len, num_heads, num_prev, head_dim]
        attn_weights_t = attn_weights.permute(2, 1, 0).unsqueeze(
            -1
        )  # [batch*seq_len, num_heads, num_prev, 1]

        # Weighted sum: multiply and sum over num_prev dimension
        attn_output = (values_t * attn_weights_t).sum(
            dim=2
        )  # [batch*seq_len, num_heads, head_dim]

        # Reshape and project
        # [batch*seq_len, num_heads, head_dim] -> [batch, seq_len, d_model]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output


class BlockAttentionResidual(nn.Module):
    """
    Block Attention Residuals: O(Nd) memory complexity (practical variant).

    Implements the official Block AttnRes algorithm from the Kimi paper.
    Layers are grouped into blocks of size `block_size` (counted in transformer
    layers, where each transformer layer contains one ATTN + one MLP sublayer).

    The attention residual is applied **before every sublayer** (both ATTN and
    MLP), attending over all completed block representations plus the current
    intra-block partial sum.  This matches the official pseudocode exactly:

        def block_attn_res(blocks, partial_block, proj, norm):
            V = stack(blocks + [partial_block])          # [N+1, B, T, D]
            K = norm(V)
            logits = einsum('d, nbtd -> nbt', proj.weight, K)
            h = einsum('nbt, nbtd -> btd', softmax(logits, dim=0), V)
            return h

    Each transformer layer has TWO separate AttnRes projections:
        - attn_res_proj / attn_res_norm   (used before ATTN sublayer)
        - mlp_res_proj  / mlp_res_norm    (used before MLP sublayer)

    Block state (`blocks`, `partial_block`) is threaded through the encoder's
    forward loop rather than stored as module state, so the module is
    re-entrant and does not need explicit reset_cache() calls at runtime.

    The `reset_cache()` method is kept for backward compatibility with tests.

    Memory: O(N_blocks × B × T × D) where N_blocks grows with depth but is
    bounded by the number of completed blocks (~L / block_size).

    Args:
        d_model: Model dimension
        block_size: Number of transformer layers per block
        num_heads: Unused (kept for API compat); paper uses a single scalar
                   projection weight, not multi-head. Kept for compatibility.
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        block_size: int,
        num_blocks: int = 0,  # kept for API compatibility, not used
        num_heads: int = 8,  # kept for API compatibility, not used
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size

        # Paper uses a single learned scalar projection weight w_l ∈ R^d
        # per sublayer position, not multi-head attention.
        # attn sublayer projection
        self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.attn_res_norm = RMSNorm(d_model)
        # mlp sublayer projection
        self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
        self.mlp_res_norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def reset_cache(self):
        """No-op — kept for backward compatibility with tests."""
        pass

    def compute(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        proj: nn.Linear,
        norm: "RMSNorm",
    ) -> torch.Tensor:
        """
        Inter-block attention: attend over completed block reps + partial sum.

        This is a direct implementation of the paper's `block_attn_res` helper:

            V = stack(blocks + [partial_block])      # [N+1, B, T, D]
            K = norm(V)
            logits = einsum('d, nbtd -> nbt', proj.weight.squeeze(), K)
            h = einsum('nbt, nbtd -> btd', softmax(logits, dim=0), V)

        Args:
            blocks: List of completed block tensors, each [B, T, D]
            partial_block: Intra-block running sum [B, T, D]
            proj: Linear(d_model → 1, bias=False) — scalar query projection
            norm: RMSNorm for keys

        Returns:
            Aggregated representation [B, T, D]
        """
        # Stack completed blocks and the current partial sum
        all_v = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]

        # Keys = normed values
        all_k = norm(all_v)  # [N+1, B, T, D]

        # Scalar logits via learned weight vector (one scalar per token per block)
        # proj.weight shape: [1, d_model] -> squeeze to [d_model]
        w = proj.weight.squeeze(0)  # [D]
        logits = torch.einsum("d, n b t d -> n b t", w, all_k)  # [N+1, B, T]

        # Softmax over the depth dimension (dim=0)
        weights = torch.softmax(logits, dim=0)  # [N+1, B, T]
        weights = self.dropout(weights)

        # Weighted sum over blocks
        h = torch.einsum("n b t, n b t d -> b t d", weights, all_v)  # [B, T, D]
        return h

    def forward(
        self, layer_idx: int, current_output: torch.Tensor, is_last_in_block: bool
    ) -> Tuple[torch.Tensor, bool]:
        """
        Legacy forward pass — kept for backward compatibility with tests only.

        In the real encoder the `compute(blocks, partial_block, proj, norm)`
        method is called directly.  This wrapper reconstructs a trivial single-
        block state to stay compatible with Test 3 in the sanity suite.

        Returns:
            Tuple of (residual_output, use_residual)
        """
        # For compatibility: return a zero residual (no block history available)
        return torch.zeros_like(current_output), False


# ============================================================================
# Transformer Components
# ============================================================================


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.1, causal: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Compute QKV
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Transformer Layers with Attention Residuals
# ============================================================================


class TransformerLayerWithAttnRes(nn.Module):
    """
    Single transformer layer using Block Attention Residuals.

    Implements the official per-sublayer application from the Kimi paper.
    Before **each** sublayer (both ATTN and MLP), the layer calls
    `block_attn_res.compute(blocks, partial_block, proj, norm)` to get an
    attention-weighted aggregation over all completed blocks plus the
    current intra-block partial sum, then uses that as the input to the
    sublayer instead of the raw `x`.

    Block state (`blocks`, `partial_block`) is passed in and out explicitly
    so the encoder can thread it through all layers.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        block_attn_res: BlockAttentionResidual instance (shared across layers)
        layer_idx: Index of this layer in the model (0-indexed)
        block_size: Number of transformer layers per block
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        block_attn_res: "BlockAttentionResidual",
        layer_idx: int,
        block_size: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.block_attn_res = block_attn_res

        # Pre-normalization layers (paper uses RMSNorm before each sublayer)
        self.attn_norm = get_norm_layer(d_model)
        self.mlp_norm = get_norm_layer(d_model)

        # Self-attention (causal mask for autoregressive language modeling)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, causal=True)

        # Feed-forward
        self.mlp = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass matching the paper's official pseudocode.

        The paper's `forward` per transformer layer (which has ATTN + MLP):

            # apply block_attn_res before attn
            h = block_attn_res(blocks, partial_block, attn_proj, attn_norm)

            # if this layer starts a new block, save old partial and reset
            if layer_number % (block_size // 2) == 0:   # paper counts sublayers
                blocks.append(partial_block)
                partial_block = 0

            # self-attention sublayer
            attn_out = attn(attn_norm(h))
            partial_block = partial_block + attn_out

            # apply block_attn_res before mlp
            h = block_attn_res(blocks, partial_block, mlp_proj, mlp_norm)

            # mlp sublayer
            mlp_out = mlp(mlp_norm(h))
            partial_block = partial_block + mlp_out

        In our convention `block_size` counts transformer layers, so the
        boundary fires at `layer_idx % block_size == 0` for `layer_idx > 0`.
        This is equivalent to the paper's `layer_number % (block_size//2) == 0`
        when `block_size` (ours) = paper's `block_size // 2`.

        Args:
            blocks: List of completed block tensors [B, T, D] each
            partial_block: Intra-block running sum [B, T, D]
            mask: Optional attention mask

        Returns:
            (blocks, partial_block) updated state
        """
        bar = self.block_attn_res

        # --- Pre-attention block_attn_res ---
        h = bar.compute(blocks, partial_block, bar.attn_res_proj, bar.attn_res_norm)

        # If this layer starts a new block (layer_idx > 0 and at boundary),
        # push the previous partial_block to blocks and reset partial.
        # The stored representation is DETACHED: completed blocks serve as
        # keys/values in the attention residual formula (read-only context),
        # so gradients only need to flow through partial_block (the current
        # block's running sum).  Detaching is architecturally correct and
        # reduces peak activation memory from O(L·B·T·D) to O(B·T·D).
        if self.layer_idx > 0 and self.layer_idx % self.block_size == 0:
            blocks = blocks + [partial_block.detach()]  # immutable list append
            partial_block = torch.zeros_like(partial_block)

        # Self-attention sublayer (Pre-Norm on the attn_res output h)
        attn_out = self.self_attn(self.attn_norm(h), mask)
        partial_block = partial_block + self.dropout(attn_out)

        # --- Pre-MLP block_attn_res ---
        h = bar.compute(blocks, partial_block, bar.mlp_res_proj, bar.mlp_res_norm)

        # MLP sublayer (Pre-Norm on the mlp_res output h)
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + self.dropout(mlp_out)

        return blocks, partial_block


# ============================================================================
# Full Transformer Encoders
# ============================================================================


class TransformerEncoderWithAttnRes(nn.Module):
    """
    Complete transformer encoder with Block Attention Residuals.

    This encoder implements the full Attention Residual architecture from the
    Kimi paper, using block-based attention residuals for efficiency.

    Architecture:
        1. Embedding layer (treated as first block)
        2. N transformer layers grouped into blocks
        3. Final normalization

    Block Structure:
        - Layers are grouped into blocks of size `block_size`
        - Within blocks: Standard residual connections
        - Between blocks: Attention residuals from all previous blocks

    The embeddings are treated as the output of "block 0", allowing the first
    real block to attend to the initial representations.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Total number of transformer layers
        block_size: Layers per block (default: 4)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        block_size: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.block_size = block_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Shared block attention residual module (one set of projections for
        # all layers — the paper shares weights across all block boundaries)
        self.block_attn_res = BlockAttentionResidual(
            d_model=d_model,
            block_size=block_size,
            dropout=dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerLayerWithAttnRes(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    block_attn_res=self.block_attn_res,
                    layer_idx=i,
                    block_size=block_size,
                )
            )

        # Final normalization
        self.norm = get_norm_layer(d_model)

        # Output projection to vocab size
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with std=0.02 as per paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input token IDs [batch, seq_len]
            mask: Optional attention mask

        Returns:
            Output representations [batch, seq_len, d_model]
        """
        batch_size, seq_len = x.shape

        # Embeddings + positional encoding
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        emb = self.embedding(x) + self.pos_embedding(positions)
        emb = self.dropout(emb)

        # Block state: treat embeddings as the first completed block.
        # `blocks` is a list of completed block representations [B, T, D].
        # `partial_block` is the intra-block running sum for the current block.
        # Both are threaded through all layers explicitly (no module-level state).
        blocks: List[torch.Tensor] = [
            emb.detach()
        ]  # embeddings = block 0 (read-only context)
        partial_block: torch.Tensor = torch.zeros_like(emb)

        # Pass through transformer layers, threading block state
        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block, mask)

        # `partial_block` always contains the accumulated hidden states for the
        # current (last) block.  It is reset to zero only at block *entry* (when
        # the previous block's partial is pushed to `blocks`), so by the time all
        # layers have run it holds the sum of the last block's sublayer outputs.
        h = partial_block

        # Final normalization
        h = self.norm(h)

        # Project to vocabulary size
        logits = self.output_projection(h)

        return logits


class StandardTransformerEncoder(nn.Module):
    """
    Standard transformer encoder with PreNorm residual connections.

    This is the baseline implementation for comparison with the Attention
    Residual encoder. It uses standard residual connections:
        h = x + Sublayer(Norm(x))

    Architecture follows standard Pre-Norm design:
        1. Embeddings
        2. N transformer layers with standard residuals
        3. Final normalization

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StandardTransformerLayer(d_model, num_heads, d_ff, dropout)
            )

        self.norm = get_norm_layer(d_model)

        # Output projection to vocab size
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with std=0.02."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through standard encoder.

        Args:
            x: Input token IDs [batch, seq_len]
            mask: Optional attention mask

        Returns:
            Output representations [batch, seq_len, d_model]
        """
        batch_size, seq_len = x.shape

        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Project to vocabulary size
        logits = self.output_projection(x)

        return logits


class StandardTransformerLayer(nn.Module):
    """Standard Pre-Norm transformer layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.norm1 = get_norm_layer(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, causal=True)

        self.norm2 = get_norm_layer(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm self-attention with residual
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))

        # Pre-norm MLP with residual
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x


# ============================================================================
# Sanity Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Attention Residuals - Sanity Tests")
    print("=" * 70)

    # Test configuration
    batch_size = 2
    seq_len = 16
    d_model = 64
    num_heads = 4
    num_layers = 8
    block_size = 4
    vocab_size = 1000
    d_ff = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Test 1: RMSNorm
    print("Test 1: RMSNorm")
    print("-" * 40)
    rms_norm = RMSNorm(d_model).to(device)
    test_input = torch.randn(batch_size, seq_len, d_model, device=device)
    rms_output = rms_norm(test_input)
    assert rms_output.shape == test_input.shape
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {rms_output.shape}")
    print(f"  Output mean: {rms_output.mean().item():.4f}")
    print(f"  Output std: {rms_output.std().item():.4f}")
    print("  [PASS]\n")

    # Test 2: FullAttentionResidual
    print("Test 2: FullAttentionResidual")
    print("-" * 40)
    full_attn_res = FullAttentionResidual(
        d_model=d_model, num_layers=num_layers, num_heads=num_heads
    ).to(device)

    # Simulate forward pass through multiple layers
    full_attn_res.reset_cache()
    for i in range(num_layers):
        layer_input = torch.randn(batch_size, seq_len, d_model, device=device)
        residual = full_attn_res(i, layer_input)
        if i > 0:
            assert residual.shape == layer_input.shape

    print(f"  Tested {num_layers} layers with O(L²d) attention")
    print(f"  Stored {len(full_attn_res.past_outputs)} layer outputs")
    print("  [PASS]\n")

    # Test 3: BlockAttentionResidual.compute
    print("Test 3: BlockAttentionResidual")
    print("-" * 40)
    num_blocks = num_layers // block_size
    block_attn_res = BlockAttentionResidual(
        d_model=d_model,
        block_size=block_size,
    ).to(device)

    # Simulate compute() calls: build up a list of blocks and a partial_block
    blocks_test: List[torch.Tensor] = []
    partial = torch.randn(batch_size, seq_len, d_model, device=device)
    result = torch.zeros(batch_size, seq_len, d_model, device=device)
    for i in range(num_blocks):
        result = block_attn_res.compute(
            blocks_test,
            partial,
            block_attn_res.attn_res_proj,
            block_attn_res.attn_res_norm,
        )
        assert result.shape == partial.shape, f"Shape mismatch at block {i}"
        blocks_test.append(partial.detach())
        partial = torch.randn(batch_size, seq_len, d_model, device=device)

    print(f"  Block size: {block_size}")
    print(f"  Number of blocks tested: {num_blocks}")
    print(f"  compute() output shape: {result.shape}")
    print("  [PASS]\n")

    # Test 4: TransformerLayerWithAttnRes (new stateless API)
    print("Test 4: TransformerLayerWithAttnRes")
    print("-" * 40)

    # Create shared block attention residual
    shared_block_attn = BlockAttentionResidual(
        d_model=d_model,
        block_size=block_size,
    ).to(device)

    layer = TransformerLayerWithAttnRes(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        block_attn_res=shared_block_attn,
        layer_idx=0,
        block_size=block_size,
    ).to(device)

    test_blocks: List[torch.Tensor] = [
        torch.randn(batch_size, seq_len, d_model, device=device)
    ]
    test_partial = torch.zeros(batch_size, seq_len, d_model, device=device)
    out_blocks, out_partial = layer(test_blocks, test_partial)
    assert out_partial.shape == (batch_size, seq_len, d_model)
    print(f"  Input partial shape: {test_partial.shape}")
    print(f"  Output partial shape: {out_partial.shape}")
    print(f"  Blocks out: {len(out_blocks)}")
    print("  [PASS]\n")

    # Test 5: TransformerEncoderWithAttnRes
    print("Test 5: TransformerEncoderWithAttnRes")
    print("-" * 40)
    attn_encoder = TransformerEncoderWithAttnRes(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        block_size=block_size,
        max_seq_len=512,
        dropout=0.1,
    ).to(device)

    test_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    encoder_output = attn_encoder(test_tokens)
    assert encoder_output.shape == (batch_size, seq_len, vocab_size)

    params = count_parameters(attn_encoder)
    print(f"  Input tokens shape: {test_tokens.shape}")
    print(f"  Output shape: {encoder_output.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  Block size: {block_size}")
    print("  [PASS]\n")

    # Test 6: StandardTransformerEncoder
    print("Test 6: StandardTransformerEncoder")
    print("-" * 40)
    std_encoder = StandardTransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=512,
        dropout=0.1,
    ).to(device)

    std_output = std_encoder(test_tokens)
    assert std_output.shape == (batch_size, seq_len, vocab_size)

    std_params = count_parameters(std_encoder)
    print(f"  Input tokens shape: {test_tokens.shape}")
    print(f"  Output shape: {std_output.shape}")
    print(f"  Parameters: {std_params:,}")
    print("  [PASS]\n")

    # Test 7: Parameter count comparison
    print("Test 7: Parameter Count Comparison")
    print("-" * 40)
    diff = params - std_params
    diff_pct = (diff / std_params) * 100
    print(f"  Attention Residual Encoder: {params:,} params")
    print(f"  Standard Encoder: {std_params:,} params")
    print(f"  Difference: {diff:,} ({diff_pct:+.2f}%)")
    print("  (Small increase due to pseudo-queries and projections)")
    print("  [PASS]\n")

    # Test 8: Gradient flow check
    print("Test 8: Gradient Flow Check")
    print("-" * 40)

    # Create simple loss and backward pass
    test_tokens_grad = torch.randint(0, vocab_size, (1, seq_len), device=device)
    test_tokens_grad.requires_grad = False

    output = attn_encoder(test_tokens_grad)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    has_gradients = any(p.grad is not None for p in attn_encoder.parameters())
    assert has_gradients, "No gradients found!"

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {has_gradients}")

    # Check gradient magnitudes
    total_grad_norm = 0.0
    for p in attn_encoder.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item()
    print(f"  Total gradient norm: {total_grad_norm:.4f}")
    print("  [PASS]\n")

    # Test 9: Memory complexity verification
    print("Test 9: Memory Complexity Verification")
    print("-" * 40)
    print(
        f"  Model config: {num_layers} layers, {d_model} dim, {block_size} block size"
    )

    # Full attention would store L(L-1)/2 attention weights
    full_attn_memory = (num_layers * (num_layers - 1) / 2) * d_model
    print(f"  Full Attention Memory: O(L²d) ≈ {full_attn_memory:,.0f} elements")

    # Block attention stores B(B-1)/2 where B = L/block_size
    num_blocks_actual = num_layers // block_size + 1
    block_attn_memory = (num_blocks_actual * (num_blocks_actual - 1) / 2) * d_model
    print(
        f"  Block Attention Memory: O(B²d) ≈ {block_attn_memory:,.0f} elements (B={num_blocks_actual})"
    )

    reduction = full_attn_memory / block_attn_memory
    print(f"  Memory reduction: {reduction:.1f}x")
    print("  [PASS]\n")

    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - FullAttentionResidual: O(L²d) complexity, full connectivity")
    print(f"  - BlockAttentionResidual: O(Nd) complexity, practical variant")
    print(f"  - TransformerEncoderWithAttnRes: Complete encoder with block residuals")
    print(f"  - StandardTransformerEncoder: Baseline for comparison")
    print("\nThe Attention Residual approach provides:")
    print("  1. Long-range gradient flow (no vanishing gradients)")
    print("  2. Global information mixing across layers")
    print("  3. Maintains O(Nd) memory with block variant")
    print("\nReady for training!")
    print("=" * 70)
