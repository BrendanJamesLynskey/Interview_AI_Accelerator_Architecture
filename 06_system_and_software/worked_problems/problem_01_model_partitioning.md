# Worked Problem: Model Partitioning

## Problem Statement

You need to deploy a 405B parameter language model (Llama 3.1 405B) for inference with the following constraints:
- Model: 126 layers, D=16384, D_ff=53248, H=128 heads, H_kv=8 (GQA), D_h=128
- Target: serve at batch size 64, sequence length 4096, using BF16 weights
- Hardware: NVIDIA H100 GPUs with 80 GB HBM each, NVLink 900 GB/s intra-node (8 GPUs), InfiniBand 400 Gb/s inter-node

Determine: (1) minimum number of GPUs, (2) optimal parallelism strategy, (3) estimated per-token latency.

---

## Worked Solution

### Step 1: Memory requirements

**Weights (BF16)**: 405B * 2 bytes = 810 GB

**KV cache per request at S=4096**:
```
Per layer: 2 * H_kv * D_h * S * 2 = 2 * 8 * 128 * 4096 * 2 = 16.78 MB per layer per request
126 layers: 126 * 16.78 = 2.11 GB per request
64 requests: 64 * 2.11 = 135 GB
```

**Total memory**: 810 + 135 = 945 GB

### Step 2: Minimum GPU count

For memory: 945 / 80 = 11.8, so 12 GPUs minimum. Use 16 GPUs (2 nodes), which leaves room for activations and fits power-of-two parallelism.

### Step 3: Parallelism strategy

With 16 GPUs across 2 nodes (8 per node with NVLink):

**Option A: TP=8, PP=2** (8-way tensor parallelism within each node, 2-way pipeline parallelism across nodes)
- Per-GPU weights: 810 / 16 = 50.6 GB
- Per-GPU KV cache: 135 / 8 = 16.9 GB (KV cache is replicated across PP stages but sharded across TP)
- Actually, with PP=2, each node handles 63 layers. KV cache per node: 64 * 63 * 16.78 MB = 67.5 GB. Per GPU: 67.5 / 8 = 8.44 GB.
- Per-GPU total: 50.6 + 8.44 = 59 GB. Fits in 80 GB.

**Option B: TP=16** (16-way tensor parallelism)
- Per-GPU weights: 810 / 16 = 50.6 GB
- Per-GPU KV cache: 135 / 16 = 8.44 GB
- Per-GPU total: 59 GB. Fits.
- Problem: TP=16 spans 2 nodes, requiring inter-node all-reduce every layer. With InfiniBand (50 GB/s), this is too slow.

Choose **Option A: TP=8, PP=2**.

### Step 4: Per-token latency estimate

Assume H100 SXM, with 3.35 TB/s of HBM bandwidth per GPU.

**Compute per step (decode, batch 64)**. Parameters per layer:
- QKV: 16384 * (16384 + 2 * 1024) = 302M
- Output projection: 16384^2 = 268M
- FFN (SwiGLU: gate, up and down): 3 * 16384 * 53248 = 2,617M
- Total: 3,187M per layer (126 layers = 401.6B; the embedding and LM head make up the rest of 405B)

```
FLOPS per layer = 2 * 64 * 3,187M = 408 GFLOPS
Total per step = 126 * 408G = 51.4 TFLOPS
```

For batch 64 this is a (64, 16384) * (16384, ...) GEMM, not a matrix-vector product, but its arithmetic intensity is still only about 64 FLOPs per BF16 weight byte — far below the H100's ridge point (~295 FLOPs/byte). Decode is therefore **memory-bound**.

Per pipeline stage (8 GPUs, 63 layers):
- Compute: 25.7 TFLOPS / (8 * 990 * 0.5 TFLOPS) = 6.5 ms
- Memory: each GPU streams its 50.6 GB of weights plus 8.46 GB of KV cache = 59.1 GB at 3.35 TB/s = 17.6 ms

**TP communication (within node)**: 2 all-reduces per layer (after attention and after the FFN), each of 64 * 16384 * 2 = 2.1 MB:
```
Per all-reduce: 2 * (7/8) * 2.1 MB / 450 GB/s (NVLink per direction) = 8.2 us
Per stage: 63 layers * 2 * 8.2 us = 1.0 ms
```
(Messages this small are in practice dominated by latency rather than bandwidth.)

**PP communication (between nodes)**:
Activation transfer per step: 64 * 16384 * 2 = 2.1 MB over InfiniBand (50 GB/s): 0.04 ms

### Step 5: Total estimated latency per token

```
Per stage: max(compute 6.5 ms, memory 17.6 ms) + TP 1.0 ms = 18.6 ms
Two stages in sequence + PP transfer: 2 * 18.6 + 0.04 = ~37 ms per decode step
Each step produces one token for each of the 64 requests:
  Per-request rate: 1000 / 37 = ~27 tokens/second
  Aggregate: 64 * 27 = ~1,700 tokens/second
```

With PP=2 and a single batch, each stage idles while the other works. Splitting the batch into two micro-batches of 32 keeps both stages busy and roughly doubles aggregate throughput, while per-request latency stays near 37 ms per token.

Optimizations: FP8/INT8 weights (halves the weight streaming that dominates each step), speculative decoding, or a larger batch (the same weight stream then serves more requests).
