# LoRA for Instruction Tuning on Tiny Alpaca Subset
# Part of optimizations series

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import os


# Tiny subset of Alpaca
def load_tiny_alpaca(max_samples=50):
    """Load a very small subset of alpaca for instruction tuning demo"""
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:50]")
    print(f"Loaded {len(dataset)} instruction examples")
    return dataset


def prepare_instruction_example(example):
    """Format instruction data"""
    if example.get("input"):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return {"text": prompt + example["output"]}


if __name__ == "__main__":
    print("Loading tiny Alpaca subset for LoRA instruction tuning...")
    dataset = load_tiny_alpaca()

    # This is a template - full training would go here
    print("Dataset ready for LoRA fine-tuning.")
    print("Next step: configure LoraConfig and train with PEFT + Trainer.")
    print("\nTo be continued in lora_instruction.ipynb for full training loop.")
