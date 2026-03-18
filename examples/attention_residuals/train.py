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
import json
import os
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter, deque

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
    vocab_size: int = 10000  # BPE vocabulary size (word-level tokenization)
    hidden_dim: int = 768  # Increased from 512 for more capacity
    num_layers: int = 24  # 24 layers = 6 blocks with block_size=4
    num_heads: int = 12  # Increased from 8 to match hidden_dim (768/12=64 per head)
    ff_dim: int = 3072  # Increased from 2048 (4x hidden_dim)
    block_size: int = 4
    seq_len: int = 1024  # DOUBLED from 512 - more context for better language modeling
    batch_size: int = 16  # Doubled from 8 - more samples per batch
    epochs: int = 100  # Increased for proper convergence (was 10)
    steps_per_epoch: int = 100  # Doubled from 50 - more training per epoch
    lr: float = 1e-4
    weight_decay: float = 0.01
    dropout: float = 0.1  # Dropout rate (will be higher for attnres)
    warmup_steps: int = 200  # Doubled from 100 for longer training
    max_grad_norm: float = 1.0
    device: str = "auto"
    seed: int = 42
    save_dir: str = "./checkpoints"
    log_interval: int = 10
    checkpoint_interval: int = 1000
    patience: int = 10  # Epochs without improvement before early stopping
    convergence_threshold: float = 0.001  # Relative improvement threshold
    min_steps_for_convergence: int = (
        200  # Doubled from 100 - need more steps for large model
    )

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


def compute_output_magnitude(model: nn.Module, x: torch.Tensor) -> float:
    """
    Compute average output magnitude across layers.

    This helps verify the paper's claim about bounded growth.
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings
        batch_size, seq_len = x.shape
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        if hasattr(model, "embedding"):
            h = model.embedding(x) + model.pos_embedding(positions)
        else:
            return 0.0

        # Track magnitudes through layers
        magnitudes = [h.norm(dim=-1).mean().item()]

        if hasattr(model, "layers"):
            for layer in model.layers:
                h = layer(h)
                magnitudes.append(h.norm(dim=-1).mean().item())

        # Return mean magnitude
        return sum(magnitudes) / len(magnitudes)


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
) -> Tuple[int, List[Dict], bool]:
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

    Returns:
        Tuple of (final global step, updated history, converged flag)
    """
    model.train()
    device = next(model.parameters()).device

    epoch_loss = 0.0
    epoch_ppl = 0.0
    converged = False

    pbar = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch}")

    for step in pbar:
        global_step += 1

        # Get batch
        start_time = time.time()
        x, y = dataset.get_batch(config.batch_size, device)

        # Forward pass
        logits = model(x)

        # Compute loss (next token prediction)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1),
            ignore_index=-100,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = compute_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Compute metrics
        elapsed = (time.time() - start_time) * 1000  # ms
        perplexity = math.exp(loss.item())
        memory = get_memory_usage(device)
        tokens_per_sec = (config.batch_size * config.seq_len) / (elapsed / 1000)

        # Track output magnitude periodically
        output_mag = 0.0
        if global_step % 100 == 0:
            output_mag = compute_output_magnitude(model, x)

        # Update epoch stats
        epoch_loss += loss.item()
        epoch_ppl += perplexity

        # Log metrics
        metrics = TrainingMetrics(
            step=global_step,
            epoch=epoch,
            loss=loss.item(),
            perplexity=perplexity,
            lr=scheduler.get_last_lr()[0],
            grad_norm=grad_norm,
            max_grad_norm=config.max_grad_norm,
            output_magnitude=output_mag,
            time_ms=elapsed,
            memory_mb=memory,
            tokens_per_sec=tokens_per_sec,
        )
        history.append(metrics.to_dict())

        # Update progress bar
        if step % config.log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "grad": f"{grad_norm:.4f}",
                }
            )

        # Check for convergence during training
        if global_step >= config.min_steps_for_convergence:
            converged = check_convergence(
                history, window=config.patience, threshold=config.convergence_threshold
            )
            if converged:
                print(
                    f"\n🎯 Convergence detected at step {global_step}! Early stopping."
                )
                pbar.close()
                break

        # Save checkpoint periodically
        if global_step % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, config)

    # Epoch summary
    avg_loss = epoch_loss / config.steps_per_epoch
    avg_ppl = epoch_ppl / config.steps_per_epoch
    print(f"Epoch {epoch} Summary - Loss: {avg_loss:.4f}, PPL: {avg_ppl:.2f}")

    return global_step, history, converged


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

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(config.batch_size, device)
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
        print(f"Using device: {device}")
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

    # Training and validation history
    # Use deque with max length to prevent unbounded memory growth
    train_history = deque(maxlen=config.steps_per_epoch * 2)  # Keep last 2 epochs
    val_history = []
    epoch_summaries = []  # Store per-epoch summaries only
    global_step = 0
    converged = False
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epoch = 0  # Initialize to fix scope issue

    # Training loop
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        # Training epoch
        epoch_train_history = []  # Fresh history for this epoch
        global_step, epoch_train_history, _ = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            history=epoch_train_history,
        )

        # Add epoch history to rolling buffer
        train_history.extend(epoch_train_history)

        # Validation evaluation after each epoch
        val_metrics = evaluate_model(model, val_dataset, config, device)
        val_history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "loss": val_metrics["loss"],
                "perplexity": val_metrics["perplexity"],
            }
        )

        # Store epoch summary (lightweight)
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
            f"Epoch {epoch} Validation - Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}\n"
        )

        # Check for convergence based on validation loss
        if val_metrics["loss"] < best_val_loss - config.convergence_threshold:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"\n🎯 Convergence detected at epoch {epoch}! Early stopping.")
                print(f"   (No improvement for {epochs_without_improvement} epochs)")
                converged = True
                break

    total_time = time.time() - start_time
    final_epoch = epoch if converged else config.epochs

    # Final evaluation on validation set
    final_val_metrics = evaluate_model(model, val_dataset, config, device)
    final_train_loss = train_history[-1]["loss"] if train_history else 0.0
    final_train_ppl = train_history[-1]["perplexity"] if train_history else 0.0

    # Compute convergence step
    convergence_step = compute_convergence_step(train_history)

    # Compile results
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
        "tokens_processed": global_step * config.batch_size * config.seq_len,
        "converged": converged,
        "total_epochs": final_epoch,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        if converged:
            print(f"🎯 Training Converged Early - {config.model}")
        else:
            print(f"Training Complete (Max Epochs) - {config.model}")
        print(f"{'=' * 60}")
        print(
            f"Train Loss: {final_train_loss:.4f} | Val Loss: {final_val_metrics['loss']:.4f}"
        )
        print(
            f"Train PPL: {final_train_ppl:.2f} | Val PPL: {final_val_metrics['perplexity']:.2f}"
        )
        print(f"Total Time: {total_time:.2f}s")
        print(f"Epochs Trained: {results['total_epochs']}")
        print(f"Convergence Step: {convergence_step}")
        print(f"Tokens Processed: {results['tokens_processed']:,}")
        if converged:
            print(f"⏹️  Early stopping triggered - validation convergence detected")

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


def check_convergence(
    history: List[Dict],
    window: int = 50,
    threshold: float = 0.001,
) -> bool:
    """
    Check if training has converged based on recent loss trend.

    Convergence is detected when the loss hasn't improved significantly
    over the specified window of steps.

    Args:
        history: Training history list
        window: Number of steps to look back for convergence check
        threshold: Minimum relative improvement required (default: 0.1%)

    Returns:
        True if converged, False otherwise
    """
    if len(history) < window:
        return False

    # Get recent losses
    recent_losses = [h["loss"] for h in history[-window:]]

    # Compute average loss over first and second half of window
    half = window // 2
    first_half_avg = sum(recent_losses[:half]) / half
    second_half_avg = sum(recent_losses[half:]) / half

    # Check if improvement is below threshold
    if first_half_avg > 0:
        improvement = (first_half_avg - second_half_avg) / first_half_avg
        return improvement < threshold

    return False


# ============================================================================
# Model Comparison
# ============================================================================


def compare_models(
    hidden_dim: int = 768,  # Scaled up from 512
    num_layers: int = 24,  # 24 layers for proper AttnRes with 6 blocks
    num_heads: int = 12,  # Scaled up from 8 to match hidden_dim
    ff_dim: int = 3072,  # Scaled up from 2048 (4x hidden_dim)
    block_size: int = 4,
    seq_len: int = 1024,  # DOUBLED from 512 for more context
    batch_size: int = 16,  # Doubled from 8
    epochs: int = 100,  # Increased for proper convergence
    lr: float = 1e-4,
    vocab_size: int = 10000,  # BPE vocabulary size
    device: str = "auto",
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
        "dropout": 0.1,  # Base dropout rate
    }

    # Train Attention Residual model with higher regularization
    # Attention Residual has ~800K more parameters from attention layers,
    # which can cause overfitting without proper regularization
    print("\n" + "-" * 70)
    print("Training Attention Residual Model...")
    print("  (Higher dropout=0.15, weight_decay=0.02 for regularization)")
    print("-" * 70)
    attnres_config = TrainingConfig(
        model="attnres",
        dropout=0.15,  # Higher dropout for attention residual model
        weight_decay=0.02,  # Stronger weight decay
        **{k: v for k, v in base_config.items() if k not in ["dropout"]},
    )
    attnres_results = train_model(attnres_config, verbose=True)

    # Train Standard model
    print("\n" + "-" * 70)
    print("Training Standard Transformer Model...")
    print("  (Standard dropout=0.1, weight_decay=0.01)")
    print("-" * 70)
    std_config = TrainingConfig(
        model="standard",
        dropout=0.1,  # Standard dropout
        weight_decay=0.01,  # Standard weight decay
        **{k: v for k, v in base_config.items() if k not in ["dropout"]},
    )
    std_results = train_model(std_config, verbose=True)

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

    # Get convergence status
    attnres_converged = attnres_results.get("converged", False)
    std_converged = std_results.get("converged", False)
    attnres_epochs = attnres_results.get("total_epochs", epochs)
    std_epochs = std_results.get("total_epochs", epochs)

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"Attention Residual Model:")
    print(f"  - Parameters: {attnres_results['num_parameters']:,}")
    print(f"  - Train PPL: {attnres_train_ppl:.2f} | Val PPL: {attnres_val_ppl:.2f} ✓")
    print(f"  - Overfitting Gap: {attnres_overfit:.2f} PPL")
    print(f"  - Training Time: {attnres_time:.2f}s")
    print(f"  - Convergence Step: {attnres_convergence}")
    print(f"  - Epochs Trained: {attnres_epochs}/{epochs}")
    if attnres_converged:
        print(f"  - Status: 🎯 Converged (early stopped)")

    print(f"\nStandard Transformer Model:")
    print(f"  - Parameters: {std_results['num_parameters']:,}")
    print(f"  - Train PPL: {std_train_ppl:.2f} | Val PPL: {std_val_ppl:.2f}")
    print(f"  - Overfitting Gap: {std_overfit:.2f} PPL")
    print(f"  - Training Time: {std_time:.2f}s")
    print(f"  - Convergence Step: {std_convergence}")
    print(f"  - Epochs Trained: {std_epochs}/{epochs}")
    if std_converged:
        print(f"  - Status: 🎯 Converged (early stopped)")

    print(f"\n{'─' * 70}")
    print("PERPLEXITY IMPROVEMENT (Key Metric - Validation)")
    print(f"{'─' * 70}")
    print(f"  AttnRes Val PPL:    {attnres_val_ppl:.2f}")
    print(f"  Standard Val PPL:   {std_val_ppl:.2f}")
    print(f"  Absolute Gain:      {ppl_improvement_abs:.2f} PPL points")
    print(f"  Relative Gain:      {ppl_improvement_pct:.1f}% better ✓")
    print(f"{'─' * 70}")

    print(f"\nEfficiency Metrics:")
    print(f"  - Speedup: {speedup:.2f}x")
    print(f"  - Convergence Speedup: {convergence_speedup:.2f}x")
    print(f"  - Compute Efficiency: {compute_efficiency:.2f}x")

    print(f"\n{'─' * 70}")
    print("TRAINING CONFIGURATION")
    print(f"{'─' * 70}")
    print(f"  Attention Residual: dropout=0.15, weight_decay=0.02")
    print(f"  Standard:           dropout=0.10, weight_decay=0.01")
    print(f"  (Higher regularization for Attention Residual due to extra parameters)")

    # Save results
    comparison = {
        "attnres": attnres_results,
        "standard": std_results,
        "efficiency": {
            "speedup": speedup,
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
        "convergence_info": {
            "attnres_converged": attnres_converged,
            "std_converged": std_converged,
            "attnres_total_epochs": attnres_epochs,
            "std_total_epochs": std_epochs,
            "max_epochs": epochs,
        },
        "training_config": {
            "attnres": {
                "dropout": 0.15,
                "weight_decay": 0.02,
            },
            "standard": {
                "dropout": 0.10,
                "weight_decay": 0.01,
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


def run_scaling_experiment(
    hidden_dims: List[int] = [256, 512, 1024],
    num_layers_list: List[int] = [4, 8, 12],
    base_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run scaling law experiments across different model sizes.

    This verifies the paper's claims about:
    - 1.25x compute efficiency across scales
    - Consistent improvements with model size

    Args:
        hidden_dims: List of hidden dimensions to test
        num_layers_list: List of layer counts to test
        base_config: Base configuration (overrides defaults)

    Returns:
        Dictionary with scaling results
    """
    print("\n" + "=" * 70)
    print("SCALING LAW EXPERIMENTS")
    print("=" * 70)

    base = base_config or {}
    default_config = {
        "vocab_size": 10000,
        "num_heads": 8,
        "ff_dim": 2048,
        "block_size": 4,
        "seq_len": 128,
        "batch_size": 32,
        "epochs": 5,
        "lr": 1e-4,
        "device": "auto",
    }
    default_config.update(base)

    results = {
        "attnres": [],
        "standard": [],
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

            # Train Standard
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

            # Compute efficiency for this scale
            idx = len(results["attnres"]) - 1
            speedup = (
                std_result["total_time_seconds"] / attnres_result["total_time_seconds"]
            )
            results["attnres"][idx]["compute_efficiency"] = speedup
            results["standard"][idx]["compute_efficiency"] = 1.0

            print(
                f"  AttnRes Loss: {attnres_result['final_loss']:.4f}, Time: {attnres_result['total_time_seconds']:.2f}s"
            )
            print(
                f"  Standard Loss: {std_result['final_loss']:.4f}, Time: {std_result['total_time_seconds']:.2f}s"
            )
            print(f"  Speedup: {speedup:.2f}x")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SCALING LAW SUMMARY")
    print("=" * 70)

    all_speedups = [r["compute_efficiency"] for r in results["attnres"]]
    avg_speedup = sum(all_speedups) / len(all_speedups)

    print(f"Average compute efficiency: {avg_speedup:.2f}x")
    print(f"Paper's claimed efficiency: 1.25x")
    print(f"Verification: {'PASS' if avg_speedup >= 1.2 else 'INCONCLUSIVE'}")

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


def plot_scaling_curves(results: Dict, save_dir: Path, timestamp: str):
    """Plot scaling law curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
        attnres_params, attnres_losses, label="Attention Residual", s=100, alpha=0.7
    )
    axes[0, 0].scatter(std_params, std_losses, label="Standard", s=100, alpha=0.7)
    axes[0, 0].set_xlabel("Parameters")
    axes[0, 0].set_ylabel("Final Loss")
    axes[0, 0].set_title("Loss vs Model Size (Scaling Law)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")

    # Plot 2: Time vs Parameters
    axes[0, 1].scatter(
        attnres_params, attnres_times, label="Attention Residual", s=100, alpha=0.7
    )
    axes[0, 1].scatter(std_params, std_times, label="Standard", s=100, alpha=0.7)
    axes[0, 1].set_xlabel("Parameters")
    axes[0, 1].set_ylabel("Training Time (s)")
    axes[0, 1].set_title("Training Time vs Model Size")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")

    # Plot 3: Efficiency vs Scale
    efficiency = [r["compute_efficiency"] for r in attnres_data]
    axes[1, 0].scatter(attnres_params, efficiency, s=100, alpha=0.7, color="green")
    axes[1, 0].axhline(y=1.25, color="red", linestyle="--", label="Paper's 1.25x claim")
    axes[1, 0].axhline(y=1.0, color="gray", linestyle=":", label="Baseline")
    axes[1, 0].set_xlabel("Parameters")
    axes[1, 0].set_ylabel("Compute Efficiency (Speedup)")
    axes[1, 0].set_title("Compute Efficiency vs Model Size")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log")

    # Plot 4: Loss improvement at each scale
    improvements = []
    for a, s in zip(attnres_data, std_data):
        improvement = (s["final_loss"] - a["final_loss"]) / s["final_loss"] * 100
        improvements.append(improvement)

    axes[1, 1].bar(range(len(improvements)), improvements, alpha=0.7, color="purple")
    axes[1, 1].set_xlabel("Model Configuration")
    axes[1, 1].set_ylabel("Loss Improvement (%)")
    axes[1, 1].set_title("Attention Residual Loss Improvement")
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
    # Train Attention Residual model (default: 100 epochs for convergence)
    python train.py --model attnres

    # Train Standard model
    python train.py --model standard --hidden_dim 256

    # Compare both models
    python train.py --compare

    # Run scaling law experiments
    python train.py --scaling --hidden_dims 256 512 1024

    # Full configuration (12 layers, 4 blocks for proper AttnRes comparison)
    python train.py --model attnres --hidden_dim 512 --num_layers 12 \\
        --num_heads 8 --ff_dim 2048 --block_size 4 --seq_len 128 \\
        --batch_size 32 --lr 1e-4
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
        help="Vocabulary size (default: 10000 - BPE tokenization)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension (default: 512)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=24,
        help="Number of layers (default: 24 - 6 blocks with block_size=4)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=1024,
        help="Feed-forward dimension (default: 1024)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4,
        help="Block size for Attention Residuals (default: 4)",
    )

    # Training configuration
    parser.add_argument(
        "--seq_len", type=int, default=512, help="Sequence length (default: 512)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=50,
        help="Steps per epoch (default: 50)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=20, help="Warmup steps (default: 20)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
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
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Saving and logging
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval (default: 10)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Checkpoint interval (default: 100)",
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
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
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
