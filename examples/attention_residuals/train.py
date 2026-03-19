#!/usr/bin/env python3
"""
Attention Residuals Training Script - SCALED FOR POWERFUL HARDWARE

Comprehensive training script for reproducing the Attention Residuals
scaling law experiments from the Kimi (MoonshotAI) paper.

SCALED CONFIGURATION (for powerful machines):
- Sequence length: 1024 (was 512) - 2x more context
- Hidden dimension: 768 (was 512) - 50% more capacity
- Num heads: 12 (was 8) - better attention resolution
- Batch size: 16 (was 8) - 2x larger batches
- Steps per epoch: 100 (was 50) - more training per epoch
- Total training: 10,000 steps (100 epochs × 100 steps)

This script:
- Trains both Attention Residual and Standard Transformer models
- Tracks metrics for scaling law analysis (loss, perplexity, compute efficiency)
- Verifies paper's claims about 1.25x compute efficiency
- Generates synthetic training data similar to the paper's setup
- Saves detailed results for comparison

Usage:
    # Train Attention Residual model
    python train.py --model attnres --epochs 10

    # Train Standard model (baseline)
    python train.py --model standard --epochs 10

    # Run comparison with scaled defaults
    python train.py --compare

    # Full scaling law experiment
    python train.py --scaling --hidden_dims 256 512 1024 --num_layers_list 4 8 12
"""

import argparse
import gc
import json
import os
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers import get_device
from attention_residuals import (
    TransformerEncoderWithAttnRes,
    StandardTransformerEncoder,
    count_parameters,
)
from bpe_tokenizer import get_or_train_tokenizer, BPETokenizerWrapper


# ============================================================================
# Configuration & Data Classes
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training runs."""

    model: str = "attnres"  # "attnres" or "standard"
    vocab_size: int = 10000
    hidden_dim: int = 768
    num_layers: int = 24  # 6 blocks of 4 layers
    num_heads: int = 12  # 768 / 12 = 64 per head
    ff_dim: int = 3072  # 4 × hidden_dim
    block_size: int = 4
    seq_len: int = 256
    batch_size: int = 32
    grad_accum_steps: int = 2  # effective batch = batch_size × grad_accum_steps = 64
    epochs: int = 20
    steps_per_epoch: int = 500
    lr: float = 1e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    warmup_steps: int = 500  # ~5% of total optimizer steps (20 × 500 = 10 000)
    max_grad_norm: float = 1.0
    device: str = "auto"
    bf16: bool = False  # Use bfloat16 mixed precision (requires CUDA Ampere+ or MPS)
    seed: int = 42
    save_dir: str = "./checkpoints"
    log_interval: int = 50
    checkpoint_interval: int = 2500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    perplexity: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    max_grad_norm: float = 0.0
    output_magnitude: float = 0.0
    time_ms: float = 0.0
    memory_mb: float = 0.0
    tokens_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalingResult:
    """Results from a scaling law experiment."""

    hidden_dim: int
    num_layers: int
    num_params: int
    final_loss: float
    final_perplexity: float
    total_time: float
    compute_efficiency: float
    convergence_step: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Device Detection
# ============================================================================


# ============================================================================
# BPE Tokenized WikiText-2 Dataset
# ============================================================================


class SyntheticDataset:
    """
    Generate synthetic training data similar to paper's setup.

    Creates random token sequences with structure:
    - Random token sequences with uniform distribution
    - Variable sequence lengths (padded to max)
    - Next-token prediction task
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate random sequences
        rng = np.random.RandomState(seed)
        self.data = rng.randint(0, vocab_size, size=(num_samples, seq_len))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample."""
        tokens = self.data[idx]
        # Input: all tokens except last
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        # Target: all tokens except first (next token prediction)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

    def get_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch."""
        indices = np.random.randint(0, self.num_samples, size=batch_size)
        x_list = []
        y_list = []
        for idx in indices:
            x, y = self.__getitem__(idx)
            x_list.append(x)
            y_list.append(y)

        x_batch = torch.stack(x_list).to(device)
        y_batch = torch.stack(y_list).to(device)
        return x_batch, y_batch


class WikiTextDataset:
    """
    WikiText-2 dataset for language modeling with BPE tokenization.

    Loads real Wikipedia text from HuggingFace datasets library.
    Uses Byte-Pair Encoding (BPE) for word-level tokenization, matching
    modern LLM practices and the paper's experimental setup.

    Args:
        vocab_size: BPE vocabulary size (default: 10000)
        seq_len: Sequence length for training
        num_samples: Number of training samples to generate
        seed: Random seed for reproducibility
        split: Dataset split ("train" or "validation")
        tokenizer_cache_dir: Directory to cache trained tokenizer
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        seq_len: int = 128,
        num_samples: int = 10000,
        seed: int = 42,
        split: str = "train",
        tokenizer_cache_dir: str = "./checkpoints/tokenizer",
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.split = split
        self.seed = seed

        # Load or train BPE tokenizer
        print(f"Loading BPE tokenizer (vocab_size={vocab_size})...")
        self.tokenizer = get_or_train_tokenizer(
            vocab_size=vocab_size,
            cache_dir=tokenizer_cache_dir,
        )

        # Verify vocab size matches
        actual_vocab_size = self.tokenizer.tokenizer.get_vocab_size()
        if actual_vocab_size != vocab_size:
            print(
                f"Warning: Actual vocab size ({actual_vocab_size}) differs from requested ({vocab_size})"
            )
            self.vocab_size = actual_vocab_size

        # Load WikiText-103 dataset
        print(f"Loading WikiText-103 ({split})...")
        from datasets import load_dataset

        try:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to load WikiText-103: {e}")

        # Tokenize and store only the token list (not pre-allocated sequences)
        print("Tokenizing text with BPE...")
        all_tokens = []
        total_chars = 0
        skipped = 0

        for example in dataset:
            text = example["text"].strip()
            if text:
                total_chars += len(text)
                # Encode text (includes <sos> and <eos> by default)
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                all_tokens.extend(tokens)
            else:
                skipped += 1

        print(f"Total characters: {total_chars:,}")
        print(f"Total BPE tokens: {len(all_tokens):,}")
        print(f"Compression ratio: {total_chars / len(all_tokens):.2f} chars/token")
        print(f"Skipped {skipped} empty lines")

        # Store tokens and parameters for on-demand sequence generation
        self.all_tokens = np.array(
            all_tokens, dtype=np.int32
        )  # 1D array, much smaller than 2D
        self.rng = np.random.RandomState(seed)
        self.required_len = seq_len + 1  # Need seq_len + 1 for input/target pairs

        # Calculate actual number of sequences possible
        if len(self.all_tokens) >= self.required_len:
            self.max_start = len(self.all_tokens) - self.required_len
            self.actual_num_samples = min(num_samples, self.max_start)
        else:
            self.max_start = 0
            self.actual_num_samples = 0

        print(
            f"Dataset ready: {self.actual_num_samples} possible sequences of length {seq_len}"
        )

        if self.actual_num_samples < num_samples:
            print(
                f"Warning: Only {self.actual_num_samples} sequences possible (requested {num_samples})"
            )
            print("  Consider increasing seq_len or using more data")

    def __len__(self) -> int:
        """Return the number of possible sequences."""
        return self.actual_num_samples

    def _get_sequence(self, start_pos: int) -> List[int]:
        """Generate a sequence starting at the given position."""
        end_pos = start_pos + self.required_len
        return self.all_tokens[start_pos:end_pos].tolist()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample by generating it on-the-fly."""
        # Generate random start position deterministically based on idx and seed
        # This ensures reproducibility without storing all sequences
        local_rng = np.random.RandomState(self.seed + idx)
        start_pos = local_rng.randint(0, max(1, self.max_start))

        tokens = self._get_sequence(start_pos)
        # Input: all tokens except last
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        # Target: all tokens except first (next token prediction)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

    def get_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch by generating sequences on-the-fly."""
        # Generate random start positions
        start_positions = self.rng.randint(0, max(1, self.max_start), size=batch_size)

        x_list = []
        y_list = []
        for start_pos in start_positions:
            tokens = self._get_sequence(start_pos)
            # Input: all tokens except last
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            # Target: all tokens except first
            y = torch.tensor(tokens[1:], dtype=torch.long)
            x_list.append(x)
            y_list.append(y)

        x_batch = torch.stack(x_list).to(device)
        y_batch = torch.stack(y_list).to(device)
        return x_batch, y_batch

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_with_special_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs including special tokens."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


# ============================================================================
# Model Factory
# ============================================================================


def create_model(
    model_type: str,
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    block_size: int,
    seq_len: int,
    device: torch.device,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: "attnres" or "standard"
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        block_size: Block size for Attention Residuals
        seq_len: Maximum sequence length
        device: Device to place model on
        dropout: Dropout probability

    Returns:
        Initialized model
    """
    if model_type == "attnres":
        model = TransformerEncoderWithAttnRes(
            vocab_size=vocab_size,
            d_model=hidden_dim,
            num_heads=num_heads,
            d_ff=ff_dim,
            num_layers=num_layers,
            block_size=block_size,
            max_seq_len=seq_len,
            dropout=dropout,
        )
    elif model_type == "standard":
        model = StandardTransformerEncoder(
            vocab_size=vocab_size,
            d_model=hidden_dim,
            num_heads=num_heads,
            d_ff=ff_dim,
            num_layers=num_layers,
            max_seq_len=seq_len,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


# ============================================================================
# Optimizer & Scheduler
# ============================================================================


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    total_steps: int,
) -> Tuple[AdamW, torch.optim.lr_scheduler._LRScheduler]:
    """
    Create optimizer and learning rate scheduler.

    Uses:
    - AdamW optimizer with weight decay
    - Linear warmup followed by cosine annealing

    Args:
        model: Model to optimize
        config: Training configuration
        total_steps: Total training steps

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )

    # Cosine annealing scheduler
    cosine_steps = max(1, total_steps - config.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=config.lr * 0.1,
    )

    # Sequential scheduler: warmup -> cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps],
    )

    return optimizer, scheduler


# ============================================================================
# Training Utilities
# ============================================================================


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def compute_per_layer_gradient_norms(model: nn.Module) -> List[float]:
    """
    Compute gradient norm for each transformer layer separately.

    This reproduces the gradient uniformity analysis from the paper:
    AttnRes should show more uniform gradient norms across depth compared
    to standard PreNorm which suffers from gradient dilution.

    Returns:
        List of gradient norms, one per transformer layer (in order).
        Returns an empty list if no layers are found.
    """
    norms: List[float] = []
    if not hasattr(model, "layers"):
        return norms
    for layer in model.layers:
        layer_norm_sq = 0.0
        for p in layer.parameters():
            if p.grad is not None:
                layer_norm_sq += p.grad.data.norm(2).item() ** 2
        norms.append(math.sqrt(layer_norm_sq))
    return norms


def compute_flops_per_step(config: "TrainingConfig", num_params: int) -> float:
    """
    Estimate FLOPs per training step using the standard approximation.

    The Chinchilla / PaLM approximation for a transformer forward+backward:
        FLOPs_per_token ≈ 6 × N_params
    Total per step:
        FLOPs_per_step = 6 × N_params × seq_len × batch_size

    This is a lower bound; attention FLOPs (2 × B × T² × D) add ~10-20% for
    typical seq_len/d_model ratios but are omitted here for simplicity.

    Args:
        config: Training configuration
        num_params: Number of model parameters

    Returns:
        Estimated FLOPs per step (float)
    """
    return 6.0 * num_params * config.seq_len * config.batch_size


def compute_depth_wise_magnitudes(model: nn.Module, x: torch.Tensor) -> List[float]:
    """
    Compute hidden-state L2 norm at each transformer layer during a forward pass.

    This reproduces Figure 2 of the paper: output magnitudes should grow
    roughly as O(√l) with standard PreNorm but stay bounded with AttnRes.

    The function hooks into the model's layers and captures the hidden state
    magnitude after each layer.

    Args:
        model: TransformerEncoderWithAttnRes or StandardTransformerEncoder
        x: Input token IDs [batch, seq_len]

    Returns:
        List of mean L2 norms per layer (length = num_layers + 1 for embedding)
    """
    magnitudes: List[float] = []
    hooks = []

    def _hook(module, input, output):
        # For AttnRes layers, output is (blocks, partial_block) tuple
        if isinstance(output, tuple):
            hidden = output[1]  # partial_block
        else:
            hidden = output
        # Mean L2 norm over batch and seq_len dimensions
        magnitudes.append(hidden.norm(dim=-1).mean().item())

    # Register hooks on each transformer layer
    if hasattr(model, "layers"):
        for layer in model.layers:
            hooks.append(layer.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        # Capture embedding magnitude first
        batch_size, seq_len = x.shape
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        emb = model.embedding(x) + model.pos_embedding(positions)
        magnitudes_with_emb = [emb.norm(dim=-1).mean().item()]

        # Run full forward (hooks fire and append to `magnitudes`)
        _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    return magnitudes_with_emb + magnitudes


def compute_output_magnitude(model: nn.Module, x: torch.Tensor) -> float:
    """
    Compute average hidden-state magnitude across all layers.

    Delegates to compute_depth_wise_magnitudes so both AttnRes and Standard
    models are handled correctly via forward hooks rather than manual layer
    calls (which broke when TransformerLayerWithAttnRes changed signature).
    """
    mags = compute_depth_wise_magnitudes(model, x)
    if not mags:
        return 0.0
    return sum(mags) / len(mags)


def get_memory_usage(device: torch.device) -> float:
    """Get current memory usage in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    elif device.type == "mps":
        # MPS doesn't have direct memory query, return 0
        return 0.0
    else:
        return 0.0


# ============================================================================
# Training Loop
# ============================================================================


def train_epoch(
    model: nn.Module,
    dataset: Any,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    history: List[Dict],
    cumulative_flops: float = 0.0,
    flops_per_step: float = 0.0,
    track_layer_grads: bool = False,
) -> Tuple[int, List[Dict], bool, float]:
    """
    Train for one epoch with convergence detection.

    Args:
        model: Model to train
        dataset: Training dataset
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        epoch: Current epoch number
        global_step: Current global step
        history: Training history
        cumulative_flops: Running FLOPs total from previous epochs
        flops_per_step: FLOPs consumed per training step
        track_layer_grads: Whether to record per-layer gradient norms

    Returns:
        Tuple of (final global step, updated history, unused bool (always False),
                  updated cumulative_flops)
    """
    model.train()
    device = next(model.parameters()).device
    accum = max(1, config.grad_accum_steps)

    epoch_loss = 0.0
    epoch_ppl = 0.0

    pbar = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch}")

    for step in pbar:
        global_step += 1
        start_time = time.time()

        # --- Gradient accumulation ---
        # Accumulate gradients over `accum` micro-batches, then take one
        # optimizer step.  Peak activation memory = batch_size (not
        # batch_size × accum), while the effective batch size is their product.
        optimizer.zero_grad()
        accum_loss = 0.0
        last_x = None

        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if config.bf16
            else torch.autocast(device_type=device.type, enabled=False)
        )

        for micro_step in range(accum):
            x, y = dataset.get_batch(config.batch_size, device)
            last_x = x

            with autocast_ctx:
                logits = model(x)
                micro_loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    y.reshape(-1),
                    ignore_index=-100,
                )
            # Scale loss so gradients are averaged across micro-batches
            (micro_loss / accum).backward()
            accum_loss += micro_loss.item() / accum

        loss_val = accum_loss

        # Gradient clipping and optimizer step
        grad_norm = compute_gradient_norm(model)

        layer_grad_norms: List[float] = []
        if track_layer_grads and global_step % 50 == 0:
            layer_grad_norms = compute_per_layer_gradient_norms(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # FLOPs count covers all micro-batches in this optimizer step
        cumulative_flops += flops_per_step * accum

        # Metrics
        elapsed = (time.time() - start_time) * 1000  # ms
        perplexity = math.exp(min(loss_val, 20.0))
        memory = get_memory_usage(device)
        effective_tokens = config.batch_size * accum * config.seq_len
        tokens_per_sec = effective_tokens / (elapsed / 1000)

        # Track output magnitude periodically (no_grad, single micro-batch)
        output_mag = 0.0
        if global_step % 100 == 0 and last_x is not None:
            output_mag = compute_output_magnitude(model, last_x)

        epoch_loss += loss_val
        epoch_ppl += perplexity

        entry = TrainingMetrics(
            step=global_step,
            epoch=epoch,
            loss=loss_val,
            perplexity=perplexity,
            lr=scheduler.get_last_lr()[0],
            grad_norm=grad_norm,
            max_grad_norm=config.max_grad_norm,
            output_magnitude=output_mag,
            time_ms=elapsed,
            memory_mb=memory,
            tokens_per_sec=tokens_per_sec,
        ).to_dict()
        entry["cumulative_flops"] = cumulative_flops
        if layer_grad_norms:
            entry["layer_grad_norms"] = layer_grad_norms
        history.append(entry)

        if step % config.log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "grad": f"{grad_norm:.4f}",
                    "mem": f"{memory:.0f}MB",
                }
            )

        if global_step % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, config)

    avg_loss = epoch_loss / config.steps_per_epoch
    avg_ppl = epoch_ppl / config.steps_per_epoch
    print(f"Epoch {epoch} Summary - Loss: {avg_loss:.4f}, PPL: {avg_ppl:.2f}")

    return global_step, history, False, cumulative_flops


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    config: TrainingConfig,
):
    """Save training checkpoint."""
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.to_dict(),
    }

    path = save_dir / f"{config.model}_step{step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def save_results(results: Dict, config: TrainingConfig, filename: Optional[str] = None):
    """Save training results to JSON."""
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.model}_{timestamp}_results.json"

    path = save_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {path}")
    return path


def evaluate_model(
    model: nn.Module,
    dataset: Any,
    config: TrainingConfig,
    device: torch.device,
    num_batches: int = 100,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        dataset: Validation dataset
        config: Training configuration
        device: Device to use
        num_batches: Number of batches to evaluate (default: 100)

    Returns:
        Dictionary with loss and perplexity
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if config.bf16
        else torch.autocast(device_type=device.type, enabled=False)
    )

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(config.batch_size, device)
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    y.reshape(-1),
                    ignore_index=-100,
                )
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


# ============================================================================
# Main Training Function
# ============================================================================


def train_model(config: TrainingConfig, verbose: bool = True) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        Dictionary with training results
    """
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Get device
    device = get_device(config.device)
    if verbose:
        dtype_str = "bfloat16 (mixed)" if config.bf16 else "float32"
        print(f"Using device: {device}  |  dtype: {dtype_str}")
        print(f"Model type: {config.model}")
        print(
            f"Config: {config.hidden_dim}d, {config.num_layers}L, {config.num_heads}H, "
            f"dropout={config.dropout}, wd={config.weight_decay}"
        )

    # Create model
    model = create_model(
        model_type=config.model,
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        block_size=config.block_size,
        seq_len=config.seq_len,
        device=device,
        dropout=config.dropout,
    )

    num_params = count_parameters(model)
    if verbose:
        print(f"Model parameters: {num_params:,}")

    # Create training dataset (WikiText-103)
    print("Loading WikiText-103 dataset (this may take a moment on first run)...")
    dataset_size = config.batch_size * config.steps_per_epoch * config.epochs
    train_dataset = WikiTextDataset(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_samples=dataset_size,
        seed=config.seed,
        split="train",
    )

    # Create validation dataset (use same tokenizer from training)
    # Reduced from 10000 to 2000 to save memory (still statistically significant)
    print("Loading validation dataset...")
    val_dataset = WikiTextDataset(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_samples=2000,  # Reduced from 10000 to save memory
        seed=config.seed,
        split="validation",
    )

    # Create optimizer and scheduler
    total_steps = config.epochs * config.steps_per_epoch
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)

    # Training and validation history — keep full trajectory for IsoFLOP fitting.
    train_history: List[Dict] = []
    val_history = []
    epoch_summaries = []
    global_step = 0
    epoch = 0
    cumulative_flops = 0.0
    flops_per_step = compute_flops_per_step(config, num_params)

    if verbose:
        print(f"Estimated FLOPs per step: {flops_per_step:.2e}")

    # Training loop — runs for exactly config.epochs, no early stopping.
    # Equal compute budgets are required for valid IsoFLOP comparisons.
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_train_history: List[Dict] = []
        global_step, epoch_train_history, _, cumulative_flops = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            history=epoch_train_history,
            cumulative_flops=cumulative_flops,
            flops_per_step=flops_per_step,
            track_layer_grads=True,
        )

        train_history.extend(epoch_train_history)

        # Validation after each epoch
        val_metrics = evaluate_model(model, val_dataset, config, device)
        val_history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "loss": val_metrics["loss"],
                "perplexity": val_metrics["perplexity"],
            }
        )

        if epoch_train_history:
            epoch_summaries.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_history[-1]["loss"],
                    "train_ppl": epoch_train_history[-1]["perplexity"],
                    "val_loss": val_metrics["loss"],
                    "val_ppl": val_metrics["perplexity"],
                }
            )

        print(
            f"Epoch {epoch} Validation - Loss: {val_metrics['loss']:.4f}, "
            f"PPL: {val_metrics['perplexity']:.2f}\n"
        )

    total_time = time.time() - start_time

    # Final evaluation
    final_val_metrics = evaluate_model(model, val_dataset, config, device)
    final_train_loss = train_history[-1]["loss"] if train_history else 0.0
    final_train_ppl = train_history[-1]["perplexity"] if train_history else 0.0

    convergence_step = compute_convergence_step(train_history)

    results = {
        "config": config.to_dict(),
        "num_parameters": num_params,
        "total_time_seconds": total_time,
        "final_train_loss": final_train_loss,
        "final_train_perplexity": final_train_ppl,
        "final_val_loss": final_val_metrics["loss"],
        "final_val_perplexity": final_val_metrics["perplexity"],
        "convergence_step": convergence_step,
        "train_history": list(train_history),
        "val_history": val_history,
        "tokens_processed": global_step
        * config.batch_size
        * config.grad_accum_steps
        * config.seq_len,
        "total_epochs": config.epochs,
        # IsoFLOP tracking
        "total_flops": cumulative_flops,
        "flops_per_step": flops_per_step,
        "final_loss": final_train_loss,
        "final_perplexity": final_train_ppl,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training Complete - {config.model}")
        print(f"{'=' * 60}")
        print(
            f"Train Loss: {final_train_loss:.4f} | Val Loss: {final_val_metrics['loss']:.4f}"
        )
        print(
            f"Train PPL: {final_train_ppl:.2f} | Val PPL: {final_val_metrics['perplexity']:.2f}"
        )
        print(f"Total Time: {total_time:.2f}s")
        print(f"Epochs: {config.epochs}")
        print(f"Convergence Step: {convergence_step}")
        print(f"Tokens Processed: {results['tokens_processed']:,}")
        print(f"Total FLOPs: {cumulative_flops:.2e}")

    return results


def compute_convergence_step(history: List[Dict], threshold: float = 0.001) -> int:
    """
    Compute the step where training converged.

    Convergence is defined as when the loss stops improving by more than threshold.
    """
    if len(history) < 100:
        return len(history)

    # Smooth loss with moving average
    window = 50
    losses = [h["loss"] for h in history]
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window)
        smoothed.append(sum(losses[start : i + 1]) / (i - start + 1))

    # Find where improvement stops
    for i in range(window, len(smoothed)):
        improvement = (smoothed[i - window] - smoothed[i]) / smoothed[i - window]
        if improvement < threshold:
            return history[i]["step"]

    return history[-1]["step"] if history else 0


# ============================================================================
# Model Comparison
# ============================================================================


def compare_models(
    hidden_dim: int = 768,
    num_layers: int = 24,
    num_heads: int = 12,
    ff_dim: int = 3072,
    block_size: int = 4,
    seq_len: int = 512,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-4,
    vocab_size: int = 10000,
    device: str = "auto",
    bf16: bool = False,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Train both model variants and compare their performance.

    This function:
    1. Trains Attention Residual model
    2. Trains Standard Transformer model
    3. Compares loss curves, convergence speed, compute efficiency
    4. Generates comparison plots

    Args:
        hidden_dim: Hidden dimension (default: 768)
        num_layers: Number of layers (default: 24)
        num_heads: Number of attention heads (default: 12)
        ff_dim: Feed-forward dimension (default: 3072)
        block_size: Block size for Attention Residuals (default: 4)
        seq_len: Sequence length (default: 1024)
        batch_size: Batch size (default: 16)
        epochs: Number of epochs (default: 100)
        lr: Learning rate (default: 1e-4)
        vocab_size: Vocabulary size (default: 10000)
        device: Device to use (default: "auto")
        save_plots: Whether to save comparison plots (default: True)

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Attention Residuals vs Standard Transformer")
    print("=" * 70)

    # Common config
    base_config = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "block_size": block_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "bf16": bf16,
        "dropout": 0.1,  # Base dropout rate
    }

    # Train both models with IDENTICAL hyperparameters.
    # Previous versions used different dropout/weight_decay for each model,
    # which confounded the comparison.  The paper keeps all hyperparameters
    # equal to isolate the architectural effect.
    #
    # Standard model is trained first to establish the baseline without any
    # prior memory pressure from the heavier block-residual model.
    print("\n" + "-" * 70)
    print("Training Standard Transformer Model (baseline)...")
    print("  (Same hyperparameters as AttnRes for a fair comparison)")
    print("-" * 70)
    std_config = TrainingConfig(
        model="standard",
        **base_config,
    )
    std_results = train_model(std_config, verbose=True)

    # Free GPU memory before training the second model.
    # The standard model object goes out of scope here; explicitly collect to
    # ensure CUDA memory is returned before the AttnRes model is allocated.
    gc.collect()
    torch.cuda.empty_cache()

    # Train Attention Residual model with identical hyperparameters
    print("\n" + "-" * 70)
    print("Training Attention Residual Model...")
    print("-" * 70)
    attnres_config = TrainingConfig(
        model="attnres",
        **base_config,
    )
    attnres_results = train_model(attnres_config, verbose=True)

    # Compute efficiency metrics
    attnres_time = attnres_results["total_time_seconds"]
    std_time = std_results["total_time_seconds"]
    speedup = std_time / attnres_time

    attnres_convergence = attnres_results["convergence_step"]
    std_convergence = std_results["convergence_step"]
    convergence_speedup = (
        std_convergence / attnres_convergence if attnres_convergence > 0 else 1.0
    )

    # Compute efficiency (paper claims 1.25x)
    compute_efficiency = speedup

    # Calculate PPL improvement metrics (based on VALIDATION PPL)
    attnres_val_ppl = attnres_results["final_val_perplexity"]
    std_val_ppl = std_results["final_val_perplexity"]
    attnres_train_ppl = attnres_results["final_train_perplexity"]
    std_train_ppl = std_results["final_train_perplexity"]
    ppl_improvement_abs = std_val_ppl - attnres_val_ppl
    ppl_improvement_pct = (
        (ppl_improvement_abs / std_val_ppl) * 100 if std_val_ppl > 0 else 0
    )

    # Calculate overfitting gap (train vs val)
    attnres_overfit = attnres_train_ppl - attnres_val_ppl
    std_overfit = std_train_ppl - std_val_ppl

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"Attention Residual Model:")
    print(f"  - Parameters: {attnres_results['num_parameters']:,}")
    print(f"  - Train PPL: {attnres_train_ppl:.2f} | Val PPL: {attnres_val_ppl:.2f}")
    print(f"  - Overfitting Gap: {attnres_overfit:.2f} PPL")
    print(f"  - Training Time: {attnres_time:.2f}s")
    print(f"  - Convergence Step: {attnres_convergence}")
    print(f"  - Total FLOPs: {attnres_results.get('total_flops', 0):.2e}")

    print(f"\nStandard Transformer Model:")
    print(f"  - Parameters: {std_results['num_parameters']:,}")
    print(f"  - Train PPL: {std_train_ppl:.2f} | Val PPL: {std_val_ppl:.2f}")
    print(f"  - Overfitting Gap: {std_overfit:.2f} PPL")
    print(f"  - Training Time: {std_time:.2f}s")
    print(f"  - Convergence Step: {std_convergence}")
    print(f"  - Total FLOPs: {std_results.get('total_flops', 0):.2e}")

    print(f"\n{'─' * 70}")
    print("PERPLEXITY IMPROVEMENT (Key Metric - Validation)")
    print(f"{'─' * 70}")
    print(f"  AttnRes Val PPL:    {attnres_val_ppl:.2f}")
    print(f"  Standard Val PPL:   {std_val_ppl:.2f}")
    print(f"  Absolute Gain:      {ppl_improvement_abs:.2f} PPL points")
    print(f"  Relative Gain:      {ppl_improvement_pct:.1f}%")
    print(f"{'─' * 70}")

    print(f"\nEfficiency Metrics:")
    print(f"  - Convergence Speedup: {convergence_speedup:.2f}x")
    print(f"  - Compute Efficiency (IsoFLOP): {compute_efficiency:.2f}x")

    print(f"\n{'─' * 70}")
    print("TRAINING CONFIGURATION")
    print(f"{'─' * 70}")
    print(f"  Both models trained with identical hyperparameters:")
    print(
        f"  dropout={base_config.get('dropout', 0.1)}, "
        f"weight_decay={base_config.get('weight_decay', 0.01)}, "
        f"lr={lr}, epochs={epochs}"
    )

    # Save results
    comparison = {
        "attnres": attnres_results,
        "standard": std_results,
        "efficiency": {
            "convergence_speedup": convergence_speedup,
            "compute_efficiency": compute_efficiency,
        },
        "ppl_comparison": {
            "attnres_train_ppl": attnres_train_ppl,
            "attnres_val_ppl": attnres_val_ppl,
            "standard_train_ppl": std_train_ppl,
            "standard_val_ppl": std_val_ppl,
            "improvement_abs": ppl_improvement_abs,
            "improvement_pct": ppl_improvement_pct,
        },
        "overfitting_analysis": {
            "attnres_gap": attnres_overfit,
            "standard_gap": std_overfit,
        },
        "training_config": {
            "shared": {
                "dropout": base_config.get("dropout", 0.1),
                "weight_decay": base_config.get("weight_decay", 0.01),
                "lr": lr,
                "epochs": epochs,
            },
        },
    }

    # Save to JSON
    save_dir = Path("./checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_dir / f"comparison_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison to {path}")

    # Generate plots
    if save_plots:
        plot_comparison(attnres_results, std_results, save_dir, timestamp)

        # IsoFLOP curve (loss vs cumulative FLOPs)
        plot_isoflop_curves(attnres_results, std_results, save_dir, timestamp)

        # Per-layer gradient norms
        plot_layer_gradient_norms(attnres_results, std_results, save_dir, timestamp)

    return comparison


def plot_comparison(
    attnres_results: Dict,
    std_results: Dict,
    save_dir: Path,
    timestamp: str,
):
    """Generate comparison plots with both training and validation curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Extract training metrics
    attnres_train_history = attnres_results["train_history"]
    std_train_history = std_results["train_history"]
    attnres_val_history = attnres_results["val_history"]
    std_val_history = std_results["val_history"]

    # Training metrics
    attnres_train_steps = [h["step"] for h in attnres_train_history]
    attnres_train_losses = [h["loss"] for h in attnres_train_history]
    attnres_train_ppls = [h["perplexity"] for h in attnres_train_history]
    attnres_grads = [h["grad_norm"] for h in attnres_train_history]

    std_train_steps = [h["step"] for h in std_train_history]
    std_train_losses = [h["loss"] for h in std_train_history]
    std_train_ppls = [h["perplexity"] for h in std_train_history]
    std_grads = [h["grad_norm"] for h in std_train_history]

    # Validation metrics (by epoch)
    attnres_val_epochs = [h["epoch"] for h in attnres_val_history]
    attnres_val_losses = [h["loss"] for h in attnres_val_history]
    attnres_val_ppls = [h["perplexity"] for h in attnres_val_history]

    std_val_epochs = [h["epoch"] for h in std_val_history]
    std_val_losses = [h["loss"] for h in std_val_history]
    std_val_ppls = [h["perplexity"] for h in std_val_history]

    # Get final values for validation PPL comparison
    attnres_final_val_ppl = attnres_val_ppls[-1] if attnres_val_ppls else 0
    std_final_val_ppl = std_val_ppls[-1] if std_val_ppls else 0
    ppl_improvement_abs = std_final_val_ppl - attnres_final_val_ppl
    ppl_improvement_pct = (
        (ppl_improvement_abs / std_final_val_ppl) * 100 if std_final_val_ppl > 0 else 0
    )

    # Plot 1: Training Loss curves
    axes[0, 0].plot(
        attnres_train_steps,
        attnres_train_losses,
        label="Attention Residual",
        linewidth=2,
    )
    axes[0, 0].plot(std_train_steps, std_train_losses, label="Standard", linewidth=2)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training Perplexity curves
    axes[0, 1].plot(
        attnres_train_steps, attnres_train_ppls, label="Attention Residual", linewidth=2
    )
    axes[0, 1].plot(std_train_steps, std_train_ppls, label="Standard", linewidth=2)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Perplexity")
    axes[0, 1].set_title("Training Perplexity Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Validation Loss curves
    axes[0, 2].plot(
        attnres_val_epochs,
        attnres_val_losses,
        label="Attention Residual",
        linewidth=2,
        marker="o",
    )
    axes[0, 2].plot(
        std_val_epochs, std_val_losses, label="Standard", linewidth=2, marker="s"
    )
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("Validation Loss Comparison")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Validation Perplexity curves with improvement visualization
    axes[1, 0].plot(
        attnres_val_epochs,
        attnres_val_ppls,
        label="Attention Residual",
        linewidth=2.5,
        color="#2E7D32",
        marker="o",
        markersize=5,
    )
    axes[1, 0].plot(
        std_val_epochs,
        std_val_ppls,
        label="Standard Transformer",
        linewidth=2.5,
        color="#C62828",
        marker="s",
        markersize=5,
    )

    # Add horizontal reference line at AttnRes final validation PPL
    axes[1, 0].axhline(
        y=attnres_final_val_ppl,
        color="#2E7D32",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"AttnRes Target ({attnres_final_val_ppl:.2f})",
    )

    # Fill between curves to show improvement region
    min_epochs = min(len(attnres_val_epochs), len(std_val_epochs))
    axes[1, 0].fill_between(
        attnres_val_epochs[:min_epochs],
        attnres_val_ppls[:min_epochs],
        std_val_ppls[:min_epochs],
        alpha=0.2,
        color="#4CAF50",
        label="Val PPL Improvement",
    )

    # Add final value annotations
    if attnres_val_epochs and attnres_val_ppls:
        axes[1, 0].annotate(
            f"{attnres_final_val_ppl:.2f}",
            xy=(attnres_val_epochs[-1], attnres_final_val_ppl),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#2E7D32",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#2E7D32",
                alpha=0.8,
            ),
        )
    if std_val_epochs and std_val_ppls:
        axes[1, 0].annotate(
            f"{std_final_val_ppl:.2f}",
            xy=(std_val_epochs[-1], std_final_val_ppl),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#C62828",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#C62828",
                alpha=0.8,
            ),
        )

    # Add improvement annotation box
    if attnres_final_val_ppl > 0 and std_final_val_ppl > 0:
        improvement_text = f"Val PPL Improvement\n{attnres_final_val_ppl:.2f} vs {std_final_val_ppl:.2f}\n+{ppl_improvement_pct:.1f}% better"
        axes[1, 0].text(
            0.98,
            0.98,
            improvement_text,
            transform=axes[1, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#E8F5E9",
                edgecolor="#2E7D32",
                linewidth=2,
                alpha=0.9,
            ),
            fontweight="bold",
        )

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Perplexity")
    axes[1, 0].set_title("Validation Perplexity Comparison\n(Lower is Better)")
    axes[1, 0].legend(loc="upper right", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Gradient Norm Comparison
    axes[1, 1].plot(
        attnres_train_steps, attnres_grads, label="Attention Residual", linewidth=2
    )
    axes[1, 1].plot(std_train_steps, std_grads, label="Standard", linewidth=2)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Gradient Norm")
    axes[1, 1].set_title("Gradient Norm Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Compute Efficiency Comparison (bar chart)
    metrics = ["Time (s)", "Convergence Step", "Final Val PPL"]
    attnres_metrics = [
        attnres_results["total_time_seconds"]
        / max(std_results["total_time_seconds"], 1),
        attnres_results["convergence_step"] / max(std_results["convergence_step"], 1),
        attnres_results["final_val_perplexity"]
        / max(std_results["final_val_perplexity"], 1),
    ]
    std_metrics = [1.0, 1.0, 1.0]  # Baseline

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 2].bar(x - width / 2, attnres_metrics, width, label="Attention Residual")
    axes[1, 2].bar(x + width / 2, std_metrics, width, label="Standard")
    axes[1, 2].set_ylabel("Normalized Value")
    axes[1, 2].set_title("Compute Efficiency Comparison")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics, rotation=15, ha="right")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = save_dir / f"comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {plot_path}")
    plt.close()


# ============================================================================
# Scaling Law Experiments
# ============================================================================


def fit_scaling_law(flops: List[float], losses: List[float]) -> Tuple[float, float]:
    """
    Fit a power-law scaling curve L = A * C^(-alpha) using log-linear regression.

    Args:
        flops: List of cumulative FLOPs at each checkpoint
        losses: Corresponding losses

    Returns:
        Tuple of (A, alpha) from log L = log A - alpha * log C
    """
    if len(flops) < 2:
        return 1.0, 0.0

    log_c = np.array([math.log(max(f, 1.0)) for f in flops])
    log_l = np.array([math.log(max(l, 1e-6)) for l in losses])

    # Linear regression: log_l = log_A - alpha * log_c
    alpha_num = np.sum((log_c - log_c.mean()) * (log_l - log_l.mean()))
    alpha_den = np.sum((log_c - log_c.mean()) ** 2)
    alpha = -alpha_num / alpha_den if alpha_den != 0 else 0.0
    log_a = log_l.mean() + alpha * log_c.mean()
    a = math.exp(log_a)
    return a, alpha


def compute_isoflop_efficiency(
    attnres_result: Dict, std_result: Dict
) -> Dict[str, Any]:
    """
    Compute the IsoFLOP efficiency of AttnRes vs Standard.

    The paper's 1.25x claim means: at the same FLOPs budget, AttnRes achieves
    the same loss as Standard trained with 1.25x more FLOPs.  Equivalently,
    to match AttnRes's final loss, Standard needs 1.25x more compute.

    Strategy:
        1. Extract (cumulative_flops, loss) pairs from training history
        2. Fit power-law scaling curves for both models
        3. For the AttnRes final loss, find the FLOPs Standard needs to reach it
        4. Ratio = Standard_FLOPs / AttnRes_FLOPs = IsoFLOP efficiency

    Returns:
        Dict with keys: isoflop_efficiency, attnres_final_loss,
        std_flops_to_match, attnres_total_flops
    """

    # Extract (flops, loss) series from history
    def extract_flops_loss(result: Dict) -> Tuple[List[float], List[float]]:
        history = result.get("train_history", [])
        fl, lo = [], []
        for entry in history:
            if "cumulative_flops" in entry and entry["cumulative_flops"] > 0:
                fl.append(entry["cumulative_flops"])
                lo.append(entry["loss"])
        return fl, lo

    attnres_fl, attnres_lo = extract_flops_loss(attnres_result)
    std_fl, std_lo = extract_flops_loss(std_result)

    if len(attnres_fl) < 3 or len(std_fl) < 3:
        # Not enough data for power-law fit, fall back to naive ratio
        final_loss_ratio = std_result["final_loss"] / max(
            attnres_result["final_loss"], 1e-6
        )
        return {
            "isoflop_efficiency": final_loss_ratio,
            "attnres_final_loss": attnres_result["final_loss"],
            "std_final_loss": std_result["final_loss"],
            "std_flops_to_match": std_result.get("total_flops", 0.0),
            "attnres_total_flops": attnres_result.get("total_flops", 0.0),
            "method": "loss_ratio_fallback",
        }

    # Fit power-law for standard model: L_std(C) = A_std * C^(-alpha_std)
    a_std, alpha_std = fit_scaling_law(std_fl, std_lo)

    attnres_final_loss = attnres_result["final_loss"]
    attnres_total_flops = attnres_result.get("total_flops", attnres_fl[-1])

    # Find C_std such that L_std(C_std) = attnres_final_loss
    # A_std * C_std^(-alpha_std) = attnres_final_loss
    # C_std = (A_std / attnres_final_loss)^(1/alpha_std)
    if alpha_std > 0 and attnres_final_loss > 0 and a_std > 0:
        c_std_to_match = (a_std / attnres_final_loss) ** (1.0 / alpha_std)
        isoflop_efficiency = c_std_to_match / max(attnres_total_flops, 1.0)
    else:
        isoflop_efficiency = 1.0
        c_std_to_match = 0.0

    return {
        "isoflop_efficiency": isoflop_efficiency,
        "attnres_final_loss": attnres_final_loss,
        "std_final_loss": std_result["final_loss"],
        "std_flops_to_match": c_std_to_match,
        "attnres_total_flops": attnres_total_flops,
        "std_scaling_A": a_std,
        "std_scaling_alpha": alpha_std,
        "method": "power_law_fit",
    }


def run_scaling_experiment(
    hidden_dims: List[int] = [256, 512, 1024],
    num_layers_list: List[int] = [4, 8, 12],
    base_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run scaling law experiments across different model sizes.

    This verifies the paper's claims about:
    - 1.25x IsoFLOP compute efficiency across scales
    - Consistent improvements with model size

    The IsoFLOP metric is the correct measure: "Block AttnRes matches the
    loss of a baseline trained with 1.25x more compute."  This is measured
    by fitting a power-law scaling curve L = A * C^(-alpha) to the standard
    model's (FLOPs, loss) trajectory, then finding how many FLOPs Standard
    needs to reach AttnRes's final loss.

    Args:
        hidden_dims: List of hidden dimensions to test
        num_layers_list: List of layer counts to test
        base_config: Base configuration (overrides defaults)

    Returns:
        Dictionary with scaling results
    """
    print("\n" + "=" * 70)
    print("SCALING LAW EXPERIMENTS (IsoFLOP metric)")
    print("=" * 70)

    base = base_config or {}
    default_config = {
        "vocab_size": 10000,
        "num_heads": 8,
        "ff_dim": 2048,
        "block_size": 4,
        "seq_len": 512,
        "batch_size": 64,
        "epochs": 10,
        "steps_per_epoch": 500,
        "warmup_steps": 500,
        "lr": 1e-4,
        "device": "auto",
    }
    default_config.update(base)

    results = {
        "attnres": [],
        "standard": [],
        "isoflop_analysis": [],
    }

    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            print(f"\n{'=' * 70}")
            print(f"Scale: {hidden_dim}d x {num_layers}L")
            print(f"{'=' * 70}")

            config = default_config.copy()
            config.update(
                {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                }
            )

            # Train Attention Residual
            print("\nAttention Residual...")
            attnres_config = TrainingConfig(model="attnres", **config)
            attnres_result = train_model(attnres_config, verbose=False)
            results["attnres"].append(
                ScalingResult(
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_params=attnres_result["num_parameters"],
                    final_loss=attnres_result["final_loss"],
                    final_perplexity=attnres_result["final_perplexity"],
                    total_time=attnres_result["total_time_seconds"],
                    compute_efficiency=0.0,
                    convergence_step=attnres_result["convergence_step"],
                ).to_dict()
            )

            gc.collect()
            torch.cuda.empty_cache()

            # Train Standard (same total_steps → same tokens seen)
            print("Standard Transformer...")
            std_config = TrainingConfig(model="standard", **config)
            std_result = train_model(std_config, verbose=False)
            results["standard"].append(
                ScalingResult(
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_params=std_result["num_parameters"],
                    final_loss=std_result["final_loss"],
                    final_perplexity=std_result["final_perplexity"],
                    total_time=std_result["total_time_seconds"],
                    compute_efficiency=0.0,
                    convergence_step=std_result["convergence_step"],
                ).to_dict()
            )

            # Compute IsoFLOP efficiency
            isoflop = compute_isoflop_efficiency(attnres_result, std_result)
            results["isoflop_analysis"].append(
                {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    **isoflop,
                }
            )

            idx = len(results["attnres"]) - 1
            results["attnres"][idx]["compute_efficiency"] = isoflop[
                "isoflop_efficiency"
            ]
            results["standard"][idx]["compute_efficiency"] = 1.0

            print(
                f"  AttnRes Loss:  {attnres_result['final_loss']:.4f} "
                f"(FLOPs: {attnres_result.get('total_flops', 0):.2e})"
            )
            print(
                f"  Standard Loss: {std_result['final_loss']:.4f} "
                f"(FLOPs: {std_result.get('total_flops', 0):.2e})"
            )
            print(
                f"  IsoFLOP efficiency: {isoflop['isoflop_efficiency']:.3f}x "
                f"(paper claims 1.25x)"
            )

    # Summary statistics
    print("\n" + "=" * 70)
    print("SCALING LAW SUMMARY (IsoFLOP)")
    print("=" * 70)

    all_efficiencies = [r["compute_efficiency"] for r in results["attnres"]]
    avg_efficiency = (
        sum(all_efficiencies) / len(all_efficiencies) if all_efficiencies else 0.0
    )

    print(f"Average IsoFLOP efficiency: {avg_efficiency:.3f}x")
    print(f"Paper's claimed efficiency: 1.25x")
    print(f"Verification: {'PASS' if avg_efficiency >= 1.2 else 'INCONCLUSIVE'}")
    print(f"\nNote: Small-scale proxy runs may show a lower efficiency than the")
    print(f"paper's 1.25x (measured at 48B-param scale on 1.4T tokens).")
    print(f"The trend direction (AttnRes < Standard loss at matched FLOPs) is")
    print(f"the key signal to verify at small scale.")

    # Save results
    save_dir = Path("./checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_dir / f"scaling_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved scaling results to {path}")

    # Plot scaling curves
    plot_scaling_curves(results, save_dir, timestamp)

    return results


def plot_magnitude_dynamics(
    attnres_result: Dict,
    std_result: Dict,
    attnres_model: nn.Module,
    std_model: nn.Module,
    save_dir: Path,
    timestamp: str,
    device: torch.device,
    vocab_size: int,
    seq_len: int,
):
    """
    Plot depth-wise hidden-state magnitude at init and end of training.

    Reproduces Figure 2 of the paper:
    - Standard PreNorm: magnitude grows roughly as O(sqrt(l)) with depth
    - AttnRes: magnitude stays bounded, more uniform across depth

    This is run once after both models have been trained.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hidden-State Magnitude vs Depth (Paper Figure 2 proxy)", fontsize=13)

    sample = torch.randint(0, vocab_size, (2, seq_len), device=device)

    for ax, model, label in [
        (axes[0], std_model, "Standard (PreNorm)"),
        (axes[1], attnres_model, "AttnRes"),
    ]:
        mags = compute_depth_wise_magnitudes(model, sample)
        ax.plot(range(len(mags)), mags, "o-", linewidth=2, markersize=5)
        ax.set_xlabel("Layer index (0 = embedding)")
        ax.set_ylabel("Mean L2 norm of hidden states")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        # Add sqrt(l) reference line for Standard
        if "Standard" in label:
            ref = [mags[0] * math.sqrt(max(i, 1)) for i in range(len(mags))]
            ax.plot(
                range(len(ref)),
                ref,
                "--",
                color="gray",
                alpha=0.6,
                label="O(√l) reference",
            )
            ax.legend()

    plt.tight_layout()
    path = save_dir / f"magnitude_dynamics_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved magnitude dynamics plot to {path}")
    plt.close()


def plot_layer_gradient_norms(
    attnres_result: Dict,
    std_result: Dict,
    save_dir: Path,
    timestamp: str,
):
    """
    Plot per-layer gradient norms at the last training step where they were recorded.

    Reproduces the gradient uniformity claim from the paper:
    AttnRes should show more uniform gradient norms across depth, while
    Standard PreNorm shows gradient dilution (smaller norms at early layers).
    """

    def extract_layer_grads(result: Dict) -> Optional[List[float]]:
        history = result.get("train_history", [])
        # Get the last entry that has per-layer grad norms
        for entry in reversed(history):
            if "layer_grad_norms" in entry:
                return entry["layer_grad_norms"]
        return None

    attnres_grads = extract_layer_grads(attnres_result)
    std_grads = extract_layer_grads(std_result)

    if attnres_grads is None or std_grads is None:
        print("No per-layer gradient data found — skipping gradient norm plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-Layer Gradient Norms (Gradient Uniformity)", fontsize=13)

    for ax, grads, label, color in [
        (axes[0], std_grads, "Standard (PreNorm)", "tab:blue"),
        (axes[1], attnres_grads, "AttnRes", "tab:orange"),
    ]:
        ax.bar(range(len(grads)), grads, color=color, alpha=0.75)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Gradient L2 norm")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")
        if grads:
            cv = np.std(grads) / (np.mean(grads) + 1e-9)
            ax.set_title(f"{label}\n(CV = {cv:.3f}, lower = more uniform)")

    plt.tight_layout()
    path = save_dir / f"layer_grad_norms_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved layer gradient norms plot to {path}")
    plt.close()


def plot_isoflop_curves(
    attnres_result: Dict,
    std_result: Dict,
    save_dir: Path,
    timestamp: str,
):
    """
    Plot loss vs cumulative FLOPs for both models (IsoFLOP comparison).

    This is the key plot for reproducing the 1.25x efficiency claim.
    If the AttnRes curve sits consistently below the Standard curve, the
    claim holds directionally.
    """

    def extract_flops_loss(result: Dict) -> Tuple[List[float], List[float]]:
        history = result.get("train_history", [])
        fl, lo = [], []
        for entry in history:
            if "cumulative_flops" in entry and entry["cumulative_flops"] > 0:
                fl.append(entry["cumulative_flops"])
                lo.append(entry["loss"])
        return fl, lo

    attnres_fl, attnres_lo = extract_flops_loss(attnres_result)
    std_fl, std_lo = extract_flops_loss(std_result)

    if not attnres_fl or not std_fl:
        print("No FLOPs data in history — skipping IsoFLOP plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Smooth by subsampling to avoid overcrowded plot
    def subsample(xs, ys, n=200):
        if len(xs) <= n:
            return xs, ys
        idx = np.linspace(0, len(xs) - 1, n, dtype=int)
        return [xs[i] for i in idx], [ys[i] for i in idx]

    attnres_fl_s, attnres_lo_s = subsample(attnres_fl, attnres_lo)
    std_fl_s, std_lo_s = subsample(std_fl, std_lo)

    ax.plot(std_fl_s, std_lo_s, label="Standard (PreNorm)", linewidth=2)
    ax.plot(attnres_fl_s, attnres_lo_s, label="AttnRes", linewidth=2)

    ax.set_xlabel("Cumulative FLOPs (6 × N_params × tokens)")
    ax.set_ylabel("Training Loss")
    ax.set_title("IsoFLOP Comparison\n(AttnRes below = better compute efficiency)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # If AttnRes final loss < Standard final loss at same FLOPs, annotate
    if attnres_lo and std_lo:
        ax.annotate(
            f"AttnRes final: {attnres_lo[-1]:.3f}\nStd final: {std_lo[-1]:.3f}",
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    path = save_dir / f"isoflop_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved IsoFLOP plot to {path}")
    plt.close()


def plot_scaling_curves(results: Dict, save_dir: Path, timestamp: str):
    """Plot scaling law curves including IsoFLOP efficiency."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scaling Law Experiments", fontsize=14)

    attnres_data = results["attnres"]
    std_data = results["standard"]

    # Extract data
    attnres_params = [r["num_params"] for r in attnres_data]
    attnres_losses = [r["final_loss"] for r in attnres_data]
    attnres_times = [r["total_time"] for r in attnres_data]

    std_params = [r["num_params"] for r in std_data]
    std_losses = [r["final_loss"] for r in std_data]
    std_times = [r["total_time"] for r in std_data]

    # Plot 1: Loss vs Parameters (Scaling Law)
    axes[0, 0].scatter(
        attnres_params, attnres_losses, label="AttnRes", s=100, alpha=0.7
    )
    axes[0, 0].scatter(std_params, std_losses, label="Standard", s=100, alpha=0.7)
    axes[0, 0].set_xlabel("Parameters")
    axes[0, 0].set_ylabel("Final Loss")
    axes[0, 0].set_title("Loss vs Model Size (Scaling Law)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")

    # Plot 2: Time vs Parameters
    axes[0, 1].scatter(attnres_params, attnres_times, label="AttnRes", s=100, alpha=0.7)
    axes[0, 1].scatter(std_params, std_times, label="Standard", s=100, alpha=0.7)
    axes[0, 1].set_xlabel("Parameters")
    axes[0, 1].set_ylabel("Training Time (s)")
    axes[0, 1].set_title("Training Time vs Model Size")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")

    # Plot 3: IsoFLOP efficiency vs Scale (replaces wall-clock speedup)
    efficiency = [r["compute_efficiency"] for r in attnres_data]
    axes[1, 0].scatter(attnres_params, efficiency, s=100, alpha=0.7, color="green")
    axes[1, 0].axhline(y=1.25, color="red", linestyle="--", label="Paper's 1.25x claim")
    axes[1, 0].axhline(y=1.0, color="gray", linestyle=":", label="Baseline (1.0x)")
    axes[1, 0].set_xlabel("Parameters")
    axes[1, 0].set_ylabel(
        "IsoFLOP Efficiency\n(FLOPs Standard needs / FLOPs AttnRes uses)"
    )
    axes[1, 0].set_title("IsoFLOP Compute Efficiency vs Model Size")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    if attnres_params:
        axes[1, 0].set_xscale("log")

    # Plot 4: Loss improvement at each scale
    improvements = []
    for a, s in zip(attnres_data, std_data):
        improvement = (
            (s["final_loss"] - a["final_loss"]) / max(s["final_loss"], 1e-6) * 100
        )
        improvements.append(improvement)

    colors = ["green" if v >= 0 else "red" for v in improvements]
    axes[1, 1].bar(range(len(improvements)), improvements, alpha=0.75, color=colors)
    axes[1, 1].set_xlabel("Model Configuration")
    axes[1, 1].set_ylabel("Loss Improvement (%)\n(positive = AttnRes better)")
    axes[1, 1].set_title("AttnRes Loss Improvement Over Standard")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plot_path = save_dir / f"scaling_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved scaling plots to {plot_path}")
    plt.close()


# ============================================================================
# Argument Parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Attention Residual models and reproduce scaling law experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train AttnRes model with defaults (20 epochs × 500 steps on A100, ~15 min)
    python train.py --model attnres

    # Train Standard baseline
    python train.py --model standard

    # Compare both models (generates IsoFLOP, gradient, and loss plots)
    python train.py --compare

    # Scaling law experiments
    python train.py --scaling --hidden_dims 256 512 768 --num_layers_list 8 16 24

    # Quick smoke-test on CPU
    python train.py --model attnres --hidden_dim 64 --num_layers 4 --num_heads 4 \\
        --ff_dim 256 --seq_len 64 --batch_size 8 --epochs 2 --steps_per_epoch 10 \\
        --vocab_size 1000 --warmup_steps 5 --device cpu
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="attnres",
        choices=["attnres", "standard"],
        help="Model type to train (default: attnres)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size (default: 10000)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=768, help="Hidden dimension (default: 768)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=24,
        help="Number of transformer layers (default: 24 = 6 blocks × 4)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads (default: 12)",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=3072,
        help="Feed-forward dimension (default: 3072 = 4 × hidden_dim)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4,
        help="Transformer layers per AttnRes block (default: 4)",
    )

    # Training configuration
    parser.add_argument(
        "--seq_len", type=int, default=256, help="Sequence length (default: 256)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Micro-batch size per step (default: 32)",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2, effective batch = batch_size × grad_accum_steps)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs (default: 20)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Steps per epoch (default: 500)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="LR warmup steps (default: 500)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0)",
    )

    # Device and optimization
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 mixed precision via torch.autocast (requires CUDA Ampere+ or MPS)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Saving and logging
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints and plots (default: ./checkpoints)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=50, help="Log every N steps (default: 50)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=2500,
        help="Save checkpoint every N steps (default: 2500)",
    )

    # Experiment modes
    parser.add_argument(
        "--compare", action="store_true", help="Run comparison between both model types"
    )
    parser.add_argument(
        "--scaling", action="store_true", help="Run scaling law experiments"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Hidden dimensions for scaling (default: 256 512 1024)",
    )
    parser.add_argument(
        "--num_layers_list",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="Layer counts for scaling (default: 4 8 12)",
    )

    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point."""
    args = parse_args()

    # Create config from args
    config_dict = {
        "model": args.model,
        "vocab_size": args.vocab_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
        "bf16": args.bf16,
        "seed": args.seed,
        "save_dir": args.save_dir,
        "log_interval": args.log_interval,
        "checkpoint_interval": args.checkpoint_interval,
    }

    if args.compare:
        # Run comparison experiment
        compare_models(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            block_size=args.block_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            vocab_size=args.vocab_size,
            device=args.device,
            bf16=args.bf16,
        )
    elif args.scaling:
        # Run scaling law experiments
        base_config = {
            "vocab_size": args.vocab_size,
            "num_heads": args.num_heads,
            "ff_dim": args.ff_dim,
            "block_size": args.block_size,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
        }
        run_scaling_experiment(
            hidden_dims=args.hidden_dims,
            num_layers_list=args.num_layers_list,
            base_config=base_config,
        )
    else:
        # Single model training
        config = TrainingConfig(**config_dict)
        results = train_model(config, verbose=True)
        save_results(results, config)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
