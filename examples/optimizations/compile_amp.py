# torch.compile + AMP for GPT-2 Inference
# Optimizes model using Torch Inductor + Automatic Mixed Precision

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time


@torch.compile(mode="reduce-overhead")
def compiled_generate(model, input_ids, max_new_tokens=20):
    """Compiled generation function"""
    with torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16
    ):
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token = outputs.logits[:, -1:].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    print("Compiling model (first run is slow)...")
    start = time.time()
    output = compiled_generate(model, inputs.input_ids, max_new_tokens=20)
    print(f"Generation took {time.time() - start:.2f}s")
    print("Output:", tokenizer.decode(output[0]))
    print("torch.compile + AMP ready for faster inference.")
