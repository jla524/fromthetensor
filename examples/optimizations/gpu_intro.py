# GPU Programming Intro for ML Inference (Inference Focused)
# Topics: CUDA concepts, matrix multiplication, fused kernels, profiling
# Run on a CUDA GPU for full experience. Uses GPT-2 small context.

import torch
import torch.nn.functional as F
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)

# ============================================================================
# 1. CUDA Basics & Memory Hierarchy
# ============================================================================


def gpu_basics():
    """Demonstrate basic CUDA memory management for inference"""
    if not torch.cuda.is_available():
        print("No CUDA. Running on CPU for demo.")
        return

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    # Example tensor (like attention weights or embeddings)
    x = torch.randn(8, 512, 768, device=device, dtype=torch.float16)
    print(
        f"Tensor shape: {x.shape}, dtype: {x.dtype}, memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    # Synchronize and timing
    torch.cuda.synchronize()
    print("CUDA memory management ready for inference workloads.")


# ============================================================================
# 2. Matrix Multiplication Optimization (Core of Attention)
# ============================================================================


def benchmark_matmul(sizes=[512, 1024, 2048], iters=50):
    """Benchmark matrix multiplication - critical for attention QKV and output proj"""
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for n in sizes:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)

        # Warmup
        for _ in range(5):
            _ = a @ b

        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()

        for _ in range(iters):
            c = a @ b

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = (time.perf_counter() - start) / iters
        results[n] = elapsed
        print(f"Matmul {n}x{n}: {elapsed * 1000:.2f} ms")

    return results


# ============================================================================
# 3. Bonus: Fused Scaled Dot Product Attention (FlashAttention style)
# ============================================================================


def benchmark_fused_attention(batch=2, seq_len=512, n_heads=12, head_dim=64):
    """Fused attention kernel via SDPA (basis of FlashAttention)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Enable FlashAttention when available
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        start = time.perf_counter()
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

    print(f"Fused Attention ({seq_len} seq_len): {latency:.2f} ms")
    return output


# ============================================================================
# 4. Load GPT-2 and demonstrate inference impact
# ============================================================================


def demo_gpt2_inference():
    """Load small GPT-2 and show where matmul/attention matter"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda().half()  # FP16 for inference speed

    input_text = "Hello, I'm a language model"
    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

    generated = tokenizer.decode(outputs[0])
    print(f"GPT-2 inference latency: {latency:.1f} ms")
    print(f"Generated: {generated[:80]}...")

    return model


print("\n=== GPU Intro Ready ===")
print(
    "Run gpu_basics(), benchmark_matmul(), benchmark_fused_attention(), and demo_gpt2_inference()"
)

if __name__ == "__main__":
    gpu_basics()
    benchmark_matmul()
    benchmark_fused_attention()
    # demo_gpt2_inference()  # uncomment when ready
