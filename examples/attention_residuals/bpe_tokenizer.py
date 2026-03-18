"""
BPE Tokenizer for Attention Residuals Implementation

This module implements Byte-Pair Encoding (BPE) tokenization using HuggingFace tokenizers.
Trained specifically on WikiText-2 dataset for language modeling experiments.

Configuration:
    - Vocabulary size: 10,000 (word-level, matches modern LLM practices)
    - Algorithm: BPE (Byte-Pair Encoding)
    - Min frequency: 2
    - Special tokens: <pad>, <unk>, <sos>, <eos>, <mask>

Usage:
    from bpe_tokenizer import BPETokenizerWrapper, get_or_train_tokenizer

    # Get or train tokenizer
    tokenizer = get_or_train_tokenizer(cache_dir="./checkpoints/tokenizer")

    # Encode text
    encoded = tokenizer.encode("Hello world")
    print(encoded.ids)  # [2, 1234, 5678, 3] (with <sos> and <eos>)

    # Decode
    text = tokenizer.decode([2, 1234, 5678, 3])
    print(text)  # "Hello world"

References:
    - HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers
    - BPE Algorithm: https://arxiv.org/abs/1508.07909
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
    from tokenizers.processors import TemplateProcessing
    from tokenizers.decoders import BPEDecoder
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        "Required dependencies not installed. "
        "Please install: pip install tokenizers datasets"
    ) from e


class BPETokenizerWrapper:
    """
    Wrapper class for BPE tokenizer with convenient encode/decode methods.

    This wrapper provides a consistent interface for encoding/decoding text
    with automatic handling of special tokens.

    Args:
        tokenizer: HuggingFace Tokenizer instance
        vocab_size: Size of the vocabulary
    """

    def __init__(self, tokenizer: Tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        self.unk_token_id = tokenizer.token_to_id("<unk>")
        self.sos_token_id = tokenizer.token_to_id("<sos>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        self.mask_token_id = tokenizer.token_to_id("<mask>")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add <sos> and <eos> tokens

        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens from output

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add <sos> and <eos> tokens

        Returns:
            List of encoded token ID lists
        """
        encodings = self.tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )
        return [enc.ids for enc in encodings]

    def save(self, path: str):
        """Save tokenizer to disk."""
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str, vocab_size: int) -> "BPETokenizerWrapper":
        """Load tokenizer from disk."""
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer, vocab_size)


def train_bpe_tokenizer(
    vocab_size: int = 10000,
    min_frequency: int = 2,
    cache_dir: str = "./checkpoints/tokenizer",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
) -> BPETokenizerWrapper:
    """
    Train a BPE tokenizer on WikiText-2 dataset.

    This function:
    1. Loads WikiText-2 from HuggingFace
    2. Trains BPE tokenizer with specified vocab size
    3. Configures special tokens and post-processing
    4. Saves tokenizer to cache for future use

    Args:
        vocab_size: Target vocabulary size (default: 10000)
        min_frequency: Minimum frequency for token inclusion (default: 2)
        cache_dir: Directory to save trained tokenizer
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration

    Returns:
        Trained BPETokenizerWrapper instance
    """
    print(f"Training BPE tokenizer on {dataset_name}...")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Min frequency: {min_frequency}")

    # Initialize BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Set up normalizer (NFD unicode normalization + lowercase + strip accents)
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    # Set up pre-tokenizer (split on whitespace)
    tokenizer.pre_tokenizer = Whitespace()

    # Set up decoder
    tokenizer.decoder = BPEDecoder()

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<unk>", "<sos>", "<eos>", "<mask>"],
        show_progress=True,
    )

    # Load WikiText-2 dataset
    print("Loading dataset for tokenizer training...")
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Prepare training data (extract non-empty text)
    def get_training_corpus():
        """Generator that yields text from dataset."""
        batch_size = 1000
        batch = []
        for example in dataset:
            text = example.get("text", "").strip()
            if text:  # Skip empty lines
                batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    # Train tokenizer
    print("Training tokenizer (this may take a few minutes)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Configure post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=[
            ("<sos>", tokenizer.token_to_id("<sos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    # Enable padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

    # Create wrapper
    wrapper = BPETokenizerWrapper(tokenizer, vocab_size)

    # Save to cache
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache_path / "tokenizer.json"
    wrapper.save(str(tokenizer_path))

    # Save metadata
    metadata = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "dataset": f"{dataset_name}/{dataset_config}",
        "actual_vocab_size": tokenizer.get_vocab_size(),
    }
    metadata_path = cache_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Tokenizer trained and saved to {tokenizer_path}")
    print(f"  Actual vocabulary size: {tokenizer.get_vocab_size()}")

    # Quick test
    test_text = "The quick brown fox jumps over the lazy dog"
    encoded = wrapper.encode(test_text)
    decoded = wrapper.decode(encoded)
    print(f"\nTest encoding:")
    print(f"  Input:  '{test_text}'")
    print(f"  Tokens: {len(encoded)} tokens")
    print(f"  Output: '{decoded}'")

    return wrapper


def get_or_train_tokenizer(
    vocab_size: int = 10000,
    min_frequency: int = 2,
    cache_dir: str = "./checkpoints/tokenizer",
    force_retrain: bool = False,
) -> BPETokenizerWrapper:
    """
    Get existing tokenizer from cache or train a new one.

    This is the main entry point for obtaining a BPE tokenizer.
    It will load from cache if available, otherwise train a new one.

    Args:
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for token inclusion
        cache_dir: Directory for tokenizer cache
        force_retrain: If True, retrain even if cache exists

    Returns:
        BPETokenizerWrapper instance
    """
    cache_path = Path(cache_dir)
    tokenizer_path = cache_path / "tokenizer.json"
    metadata_path = cache_path / "metadata.json"

    # Check if cached tokenizer exists and matches config
    if not force_retrain and tokenizer_path.exists() and metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Verify vocab size matches
            if metadata.get("vocab_size") == vocab_size:
                print(f"Loading cached tokenizer from {tokenizer_path}")
                return BPETokenizerWrapper.load(str(tokenizer_path), vocab_size)
            else:
                print(
                    f"Cached tokenizer has vocab_size={metadata.get('vocab_size')}, "
                    f"but requested {vocab_size}. Retraining..."
                )
        except Exception as e:
            print(f"Failed to load cached tokenizer: {e}. Retraining...")

    # Train new tokenizer
    return train_bpe_tokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        cache_dir=cache_dir,
    )


def calculate_compression_ratio(
    tokenizer: BPETokenizerWrapper, texts: List[str]
) -> float:
    """
    Calculate the compression ratio of BPE tokenization.

    Compression ratio = characters / tokens
    Higher is better (fewer tokens per character).

    Args:
        tokenizer: BPE tokenizer
        texts: List of texts to evaluate

    Returns:
        Compression ratio (characters per token)
    """
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
    )

    if total_tokens == 0:
        return 0.0

    return total_chars / total_tokens


if __name__ == "__main__":
    # Test the tokenizer
    print("Testing BPE Tokenizer Implementation\n")

    # Train or load tokenizer
    tokenizer = get_or_train_tokenizer(vocab_size=10000)

    # Test encoding/decoding
    test_sentences = [
        "Hello world",
        "This is a test of the BPE tokenizer.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models use attention mechanisms.",
        "Residual connections help with gradient flow.",
    ]

    print("\nEncoding/Decoding Tests:")
    print("-" * 60)

    for text in test_sentences:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Input:  '{text}'")
        print(f"Tokens: {encoded}")
        print(f"Output: '{decoded}'")
        print(f"Length: {len(encoded)} tokens")
        print("-" * 60)

    # Calculate compression ratio
    compression = calculate_compression_ratio(tokenizer, test_sentences)
    print(f"\nCompression Ratio: {compression:.2f} characters per token")
