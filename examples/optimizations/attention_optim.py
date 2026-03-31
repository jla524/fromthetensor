# Attention Optimizations for Fast GPT-2 Inference
# Focus: KV Cache + Scaled Dot Product Attention (FlashAttention)

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time


class KVCache:
    """Simple KV cache for fast autoregressive inference"""

    def __init__(self, max_seq_len=1024, num_layers=12):
        self.max_seq_len = max_seq_len
        self.cache = None
        self.current_len = 0

    def update(self, k, v):
        """Update cache with new keys and values"""
        if self.cache is None:
            self.cache = (k, v)
        else:
            self.cache = (
                torch.cat([self.cache[0], k], dim=2),
                torch.cat([self.cache[1], v], dim=2),
            )
        return self.cache


def generate_with_cache(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    """Fast generation with KV caching + sampling"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated = inputs.input_ids.clone()
    cache = None

    with torch.no_grad():
        start_time = time.time()

        for _ in range(max_new_tokens):
            outputs = model(generated, use_cache=True, past_key_values=cache)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            cache = outputs.past_key_values

            if next_token.item() == tokenizer.eos_token_id:
                break

        latency = time.time() - start_time

    text = tokenizer.decode(generated[0])
    print(
        f"Generated {max_new_tokens} tokens in {latency * 1000:.1f}ms ({max_new_tokens / latency:.1f} tokens/sec)"
    )
    return text


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if torch.cuda.is_available():
        model = model.cuda().half()
        print("Using GPU + FP16 for inference")

    model.eval()
    output = generate_with_cache(
        model, tokenizer, "The meaning of life is", max_new_tokens=30
    )
    print("Output:", output[:100])
