# Worked Problem: Memory Bandwidth Calculation

## Problem Statement

You are designing the memory subsystem for a new AI inference accelerator. The target workload is a 7-billion parameter transformer model running at batch size 1 with FP16 weights, generating tokens autoregressively. The model has:
- 32 layers, model dimension D = 4096, FFN dimension D_ff = 11008 (LLaMA-style)
- 32 attention heads, head dimension 128
- Grouped-Query Attention with 8 KV heads

Target: 50 tokens/second output generation.

Calculate:
1. Weight memory per token (all parameters loaded once per token)
2. KV cache read per token at sequence length 2048
3. Total memory bandwidth required
4. Whether a single chip with 2 TB/s HBM bandwidth can meet the requirement
5. The effect of INT8 weight quantization

---

## Worked Solution

### Step 1: Calculate model weight size

For each transformer layer, the parameters are:

**QKV projection**: Q has (D, D) = (4096, 4096), K has (D, D_kv) = (4096, 8*128) = (4096, 1024), V same as K.
```
QKV params = 4096 * 4096 + 4096 * 1024 + 4096 * 1024 = 16.78M + 4.19M + 4.19M = 25.17M
```

**Output projection**: (D, D) = (4096, 4096)
```
Output params = 16.78M
```

**FFN**: Up projection (D, D_ff), gate projection (D, D_ff), down projection (D_ff, D)
```
FFN params = 4096 * 11008 + 4096 * 11008 + 11008 * 4096 = 3 * 45.09M = 135.27M
```

**Layer norms**: Negligible (2 * D = 8192 per layer)

**Per-layer total**: 25.17M + 16.78M + 135.27M = 177.22M parameters

**All layers**: 32 * 177.22M = 5,671M parameters

**Embeddings + output head**: Vocabulary ~32000 * 4096 = 131M parameters each; LLaMA-style models do not tie them, so 262M

**Total**: approximately 5,933M = 5.9B parameters. This does not match the 7B label: the GQA configuration above has fewer attention parameters than LLaMA-2 7B, which uses full multi-head attention (32 KV heads) and has 6.74B parameters. We will use 7B as stated, which makes the bandwidth estimates below slightly conservative.

**Weight bytes (FP16)**: 7 * 10^9 * 2 = 14 GB

### Step 2: Weight bandwidth per token

For batch-1 autoregressive inference, each token generation requires a forward pass through all layers. Each linear layer is a matrix-vector multiply that loads the entire weight matrix from HBM:

```
Weight bandwidth per token = 14 GB per token
```

At 50 tokens/second:
```
Weight bandwidth = 14 GB * 50 = 700 GB/s
```

### Step 3: KV cache size and bandwidth

At sequence length S = 2048 with 8 KV heads and head dimension 128:

```
KV cache per layer = 2 * S * 8 * 128 * 2 bytes = 2 * 2048 * 8 * 128 * 2 = 8.39 MB
KV cache total = 32 * 8.39 MB = 268 MB
```

For each token, the attention mechanism reads the entire KV cache (to compute attention scores and aggregate values):
```
KV cache bandwidth per token = 268 MB per token
```

At 50 tokens/second:
```
KV cache bandwidth = 268 MB * 50 = 13.4 GB/s
```

### Step 4: Total bandwidth requirement

```
Total bandwidth = Weight bandwidth + KV cache bandwidth + Activation bandwidth
                = 700 + 13.4 + ~10 (activations, negligible)
                = ~723 GB/s
```

The KV cache is less than 2% of total bandwidth -- weights dominate at batch size 1.

### Step 5: Check against 2 TB/s HBM

```
Required: 723 GB/s
Available: 2000 GB/s (2 TB/s)
Headroom: 2000 / 723 = 2.77x
```

A single chip with 2 TB/s can comfortably meet the 50 tokens/second target. The theoretical maximum token rate is:
```
Max tokens/s = 2000 / 14.46 = ~138 tokens/second
```

(Where 14.46 GB = 14 GB weights + 0.268 GB KV cache + ~0.2 GB activations per token.)

### Step 6: Effect of INT8 weight quantization

With INT8 weights, the weight data halves:
```
Weight bytes (INT8) = 7 * 10^9 * 1 = 7 GB
Weight bandwidth at 50 tok/s = 7 * 50 = 350 GB/s
Total bandwidth = 350 + 13.4 + ~5 = ~368 GB/s
```

Maximum token rate with INT8:
```
Max tokens/s = 2000 / 7.37 = ~271 tokens/second
```

(Where 7.37 GB = 7 GB weights + 0.268 GB KV cache + ~0.1 GB activations per token.)

INT8 quantization nearly doubles the achievable token rate by halving the dominant bandwidth cost (weight loading). This is why quantization is so impactful for inference.

### Step 7: Summary

| Metric | FP16 | INT8 |
|---|---|---|
| Weight data per token | 14 GB | 7 GB |
| KV cache per token (S=2048) | 268 MB | 268 MB |
| Bandwidth at 50 tok/s | 723 GB/s | 368 GB/s |
| Max tok/s on 2 TB/s chip | 138 | 271 |
| Bandwidth utilization at 50 tok/s | 36% | 18% |

The analysis confirms that batch-1 LLM inference is dominated by weight loading and is fundamentally memory-bandwidth-bound. Increasing batch size (to amortize weight loads) and weight quantization are the two most effective optimizations.
