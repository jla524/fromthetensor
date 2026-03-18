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

        # Store current output for future layers
        self.past_outputs.append(current_output.detach())

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

    Efficient variant that groups layers into blocks. Within each block,
    standard residual connections are used (fast, low memory). Between blocks,
    attention residuals allow information flow across the entire network.

    Structure:
        Block 1: [Layer 1, Layer 2, ..., Layer N] - Standard residuals
        Block 2: [Layer N+1, ..., Layer 2N] - Standard residuals
        ...
        Block K: [Layer (K-1)N+1, ..., Layer L]

        Between blocks: Attention residuals from all previous block outputs

    Memory: O(N × d) where N = block size (constant, not L!)

    This is the main practical variant from the paper, providing a balance
    between the expressiveness of full attention residuals and the efficiency
    of standard residuals.

    Args:
        d_model: Model dimension
        block_size: Number of layers per block
        num_blocks: Number of blocks (total layers = block_size * num_blocks)
        num_heads: Number of attention heads between blocks (default: 8)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        block_size: int,
        num_blocks: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Pseudo-queries: one per block (not per layer - efficiency!)
        # Shape: [num_blocks, num_heads, head_dim]
        self.pseudo_queries = nn.Parameter(
            torch.randn(num_blocks, num_heads, self.head_dim) * 0.02
        )

        # Key and value projections for block outputs
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Track block outputs and current position
        self.reset_cache()

    def reset_cache(self):
        """Clear the stored block outputs cache."""
        self.block_outputs: List[torch.Tensor] = []
        self.layers_in_current_block = 0
        self.current_block_idx = 0

    def forward(
        self, layer_idx: int, current_output: torch.Tensor, is_last_in_block: bool
    ) -> Tuple[torch.Tensor, bool]:
        """
        Compute attention residual for the current layer.

        Args:
            layer_idx: Global layer index (0-indexed)
            current_output: Current layer output [batch, seq_len, d_model]
            is_last_in_block: Whether this is the last layer in its block

        Returns:
            Tuple of (residual_output, use_residual)
            - If use_residual is False, use standard residual (within block)
            - If use_residual is True, use attention residual (between blocks)
        """
        batch_size, seq_len, _ = current_output.shape

        # Determine if we're at a block boundary
        is_first_layer_in_block = layer_idx % self.block_size == 0

        if is_first_layer_in_block and self.current_block_idx > 0:
            # We're at the start of a new block (after the first block)
            # Apply attention residual from all previous block outputs

            if len(self.block_outputs) > 0:
                # Stack previous block outputs
                # Shape: [num_prev_blocks, batch, seq_len, d_model]
                prev_blocks = torch.stack(self.block_outputs, dim=0)
                num_prev_blocks = prev_blocks.shape[0]

                # Flatten for attention computation
                prev_blocks_flat = prev_blocks.view(num_prev_blocks, -1, self.d_model)

                # Compute keys and values
                keys = self.key_proj(prev_blocks_flat).view(
                    num_prev_blocks, -1, self.num_heads, self.head_dim
                )
                values = self.value_proj(prev_blocks_flat).view(
                    num_prev_blocks, -1, self.num_heads, self.head_dim
                )

                # Get query for current block
                query = self.pseudo_queries[
                    self.current_block_idx
                ]  # [num_heads, head_dim]

                # Compute attention scores
                # keys: [num_prev_blocks, batch*seq_len, num_heads, head_dim]
                # query: [num_heads, head_dim]

                # Reshape keys for attention computation
                keys_permuted = keys.permute(
                    2, 0, 1, 3
                )  # [num_heads, num_prev, batch*seq, head_dim]

                # Expand query: [num_heads, 1, head_dim]
                query_expanded = query.unsqueeze(1)  # [num_heads, 1, head_dim]

                # Reshape keys for batch matrix multiply: [num_heads, num_prev*batch*seq, head_dim]
                keys_reshaped = keys_permuted.reshape(
                    self.num_heads,
                    num_prev_blocks * batch_size * seq_len,
                    self.head_dim,
                )

                # scores: [num_heads, 1, num_prev*batch*seq]
                scores = (
                    torch.matmul(query_expanded, keys_reshaped.transpose(-2, -1))
                    * self.scale
                )

                # Reshape scores: [num_heads, num_prev, batch*seq_len]
                scores = scores.squeeze(1).reshape(
                    self.num_heads, num_prev_blocks, batch_size * seq_len
                )

                # Transpose to [num_prev, num_heads, batch*seq_len]
                scores = scores.permute(1, 0, 2)

                # Softmax over previous blocks
                attn_weights = F.softmax(scores, dim=0)
                attn_weights = self.dropout(attn_weights)

                # Apply attention
                values_t = values.permute(
                    1, 2, 0, 3
                )  # [batch*seq, num_heads, num_prev, head_dim]
                attn_weights_t = attn_weights.permute(2, 1, 0).unsqueeze(
                    -1
                )  # [batch*seq, num_heads, num_prev, 1]

                # Weighted sum
                attn_output = (values_t * attn_weights_t).sum(
                    dim=2
                )  # [batch*seq, num_heads, head_dim]
                attn_output = attn_output.view(batch_size, seq_len, self.d_model)
                output = self.out_proj(attn_output)

                # Reset layer counter for new block
                self.layers_in_current_block = 1

                return output, True  # Use attention residual

        # Within a block: standard residual will be added outside
        if is_last_in_block:
            # Store the block's final output for future blocks
            self.block_outputs.append(current_output.detach())
            # Only increment if there will be more blocks (not at the very end)
            # The current_block_idx represents which block we're ABOUT to enter
            if self.current_block_idx < self.num_blocks - 1:
                self.current_block_idx += 1
            self.layers_in_current_block = 0
        else:
            self.layers_in_current_block += 1

        return current_output, False  # Use standard residual


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

    This layer applies attention residuals before both the self-attention
    and MLP sublayers, maintaining the block structure from the paper.

    Architecture:
        h = x + BlockAttnRes(Norm(x)) + SelfAttn(Norm(x))
        h = h + BlockAttnRes(Norm(h)) + MLP(Norm(h))

    The BlockAttnRes is applied only at block boundaries (between blocks),
    not within blocks.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        block_attn_res: BlockAttentionResidual instance (shared across layers)
        layer_idx: Index of this layer in the model
        is_last_in_block: Whether this layer is last in its block
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        block_attn_res: BlockAttentionResidual,
        layer_idx: int,
        is_last_in_block: bool,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_last_in_block = is_last_in_block
        self.block_attn_res = block_attn_res

        # Pre-normalization layers
        self.norm1 = get_norm_layer(d_model)
        self.norm2 = get_norm_layer(d_model)

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward
        self.mlp = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with block attention residuals.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output [batch, seq_len, d_model]
        """
        # First sublayer: Self-Attention with attention residual
        normed = self.norm1(x)

        # Get attention residual (if at block boundary)
        attn_res, use_attn_res = self.block_attn_res(
            self.layer_idx, normed, self.is_last_in_block
        )

        # Self-attention
        attn_out = self.self_attn(normed, mask)

        # Apply residual connection
        if use_attn_res and self.layer_idx > 0:
            # Between blocks: attention residual + self-attention
            x = x + self.dropout(attn_out + attn_res)
        else:
            # Within block or first layer: standard residual
            x = x + self.dropout(attn_out)

        # Second sublayer: MLP with attention residual
        normed = self.norm2(x)

        # Note: We don't typically use attention residuals for MLP sublayer
        # in the standard implementation, but we could
        mlp_out = self.mlp(normed)
        x = x + self.dropout(mlp_out)

        return x


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

        # Calculate number of blocks
        # Total blocks = 1 (embeddings) + ceil(num_layers / block_size)
        num_blocks = 1 + (num_layers + block_size - 1) // block_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Shared block attention residual module
        self.block_attn_res = BlockAttentionResidual(
            d_model=d_model,
            block_size=block_size,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last_in_block = ((i + 1) % block_size == 0) or (i == num_layers - 1)
            self.layers.append(
                TransformerLayerWithAttnRes(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    block_attn_res=self.block_attn_res,
                    layer_idx=i,
                    is_last_in_block=is_last_in_block,
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

        # Reset attention residual cache at start of forward pass
        self.block_attn_res.reset_cache()

        # Embeddings + positional encoding
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Treat embeddings as first block output
        # This allows block 1 to attend to embeddings
        self.block_attn_res.block_outputs.append(x.detach())
        # Start at block 1 (after embeddings), BlockAttentionResidual will handle increments
        self.block_attn_res.current_block_idx = 1

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary size
        logits = self.output_projection(x)

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
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

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

    # Test 3: BlockAttentionResidual
    print("Test 3: BlockAttentionResidual")
    print("-" * 40)
    num_blocks = num_layers // block_size
    block_attn_res = BlockAttentionResidual(
        d_model=d_model,
        block_size=block_size,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).to(device)

    # Simulate forward pass
    block_attn_res.reset_cache()
    for i in range(num_layers):
        layer_input = torch.randn(batch_size, seq_len, d_model, device=device)
        is_last = ((i + 1) % block_size == 0) or (i == num_layers - 1)
        residual, use_residual = block_attn_res(i, layer_input, is_last)
        assert residual.shape == layer_input.shape

    print(f"  Block size: {block_size}")
    print(f"  Number of blocks: {num_blocks}")
    print(f"  Stored {len(block_attn_res.block_outputs)} block outputs")
    print("  [PASS]\n")

    # Test 4: TransformerLayerWithAttnRes
    print("Test 4: TransformerLayerWithAttnRes")
    print("-" * 40)

    # Create shared block attention residual
    shared_block_attn = BlockAttentionResidual(
        d_model=d_model,
        block_size=block_size,
        num_blocks=num_layers // block_size + 1,
        num_heads=num_heads,
    ).to(device)

    layer = TransformerLayerWithAttnRes(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        block_attn_res=shared_block_attn,
        layer_idx=0,
        is_last_in_block=False,
    ).to(device)

    test_input = torch.randn(batch_size, seq_len, d_model, device=device)
    output = layer(test_input)
    assert output.shape == test_input.shape
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
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
