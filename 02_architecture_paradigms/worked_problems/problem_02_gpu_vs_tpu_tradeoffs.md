# Worked Problem: GPU vs TPU Tradeoffs

## Problem Statement

Your team needs to train a 13-billion parameter transformer language model on a dataset of 500 billion tokens. You have two options:

**Option A: GPU Cluster**
- 64 NVIDIA H100 GPUs (8 nodes, 8 GPUs each)
- 990 TFLOPS FP16 tensor core peak per GPU, 50% sustained utilization on training
- 80 GB HBM3 per GPU at 3.35 TB/s
- 900 GB/s NVLink intra-node, 400 Gb/s InfiniBand inter-node
- $3.50/GPU-hour cloud pricing

**Option B: TPU Pod**
- 64 TPU v4 chips (a single pod slice)
- 275 TFLOPS BF16 peak per chip, 60% sustained utilization on training
- 32 GB HBM2 per chip at 1.2 TB/s
- ICI interconnect: 4800 Gb/s per chip (all-to-all via 3D torus)
- $2.50/chip-hour cloud pricing

Compare:
1. Total sustained compute throughput
2. Total training time estimate
3. Communication bottleneck analysis
4. Total memory capacity and model fit
5. Total training cost

---

## Worked Solution

### Step 1: Calculate total sustained compute

**GPU Cluster:**
```
Sustained per GPU = 990 * 0.50 = 495 TFLOPS
Total sustained = 64 * 495 = 31,680 TFLOPS
```

**TPU Pod:**
```
Sustained per chip = 275 * 0.60 = 165 TFLOPS
Total sustained = 64 * 165 = 10,560 TFLOPS
```

The GPU cluster provides 3.0x more sustained compute. Note that the TPU's higher utilization (60% vs 50%) partially compensates for lower peak throughput, but the H100's raw performance advantage is substantial.

### Step 2: Estimate total training compute

Using the 6ND approximation:
```
C = 6 * N * D = 6 * 13 * 10^9 * 500 * 10^9 = 3.9 * 10^22 FLOPS
```

### Step 3: Estimate training time

**GPU Cluster:**
```
Time = 3.9 * 10^22 / (31,680 * 10^12) = 1.231 * 10^6 seconds = 14.24 days
```

**TPU Pod:**
```
Time = 3.9 * 10^22 / (10,560 * 10^12) = 3.693 * 10^6 seconds = 42.74 days
```

### Step 4: Communication bottleneck analysis

For data-parallel training with 64 accelerators, each step requires an all-reduce of the gradient tensor (same size as the model: 13B parameters).

**Gradient size (BF16):**
```
Gradient bytes = 13 * 10^9 * 2 = 26 GB
```

**GPU cluster all-reduce time:**

Ring all-reduce with a hierarchical approach:
- Intra-node (8 GPUs, NVLink 900 GB/s bidirectional): 
  ```
  Time_intra = 2 * (7/8) * 26 GB / 900 GB/s = 0.040 s = 40 ms
  ```
- Inter-node (8 nodes, InfiniBand 400 Gb/s = 50 GB/s bidirectional):
  ```
  Time_inter = 2 * (7/8) * (26/8) GB / 50 GB/s = 0.114 s = 114 ms
  ```
  (After intra-node reduce-scatter, each GPU holds 26/8 = 3.25 GB to communicate inter-node)
- Total all-reduce: approximately 40 + 114 = 154 ms

**TPU pod all-reduce time:**

3D torus with 4800 Gb/s = 600 GB/s per chip:
- Using a 3D decomposition (4x4x4 torus), the all-reduce decomposes into three 1D ring all-reduces:
  ```
  Per-dimension all-reduce = 2 * (3/4) * 26 GB / 600 GB/s = 0.065 s = 65 ms
  ```
  (Approximate, assuming balanced 3D decomposition)
- Total: approximately 65 ms (dimensions can be pipelined)

**Compute time per step** (assuming global batch of 4M tokens, micro-batch 64K tokens per device):
```
FLOPS per step = 6 * 13 * 10^9 * 4 * 10^6 = 3.12 * 10^17 FLOPS
```

GPU: 3.12 * 10^17 / (31,680 * 10^12) = 9.85 seconds per step
TPU: 3.12 * 10^17 / (10,560 * 10^12) = 29.55 seconds per step

**Communication overhead:**
- GPU: 154 ms / 9.85 s = 1.6% (easily overlapped with backward pass)
- TPU: 65 ms / 29.55 s = 0.2% (negligible)

Both systems can effectively overlap communication with computation, so communication is not a bottleneck at 64-device scale for this model size.

### Step 5: Memory capacity analysis

**Model memory requirements (mixed-precision training):**
- Parameters (FP16): 13B * 2 = 26 GB
- Gradients (FP16): 13B * 2 = 26 GB  
- Optimizer states (FP32 master weights + momentum + variance for Adam): 13B * 12 = 156 GB
- Total (without activation memory): 208 GB

**GPU cluster:** 64 * 80 GB = 5,120 GB total. With ZeRO-3 (partitioning parameters, gradients, and optimizer states across all GPUs): 208 / 64 = 3.25 GB per GPU. Leaves approximately 77 GB per GPU for activations. Sufficient.

**TPU pod:** 64 * 32 GB = 2,048 GB total. With full sharding: 208 / 64 = 3.25 GB per chip. Leaves approximately 29 GB per chip for activations. Tighter but workable with activation checkpointing.

The GPU cluster has 2.5x more total memory capacity, providing more room for larger batch sizes or less aggressive activation checkpointing.

### Step 6: Total training cost

**GPU Cluster:**
```
Cost = 64 GPUs * 14.24 days * 24 hours/day * $3.50/GPU-hour = $76,800
```

**TPU Pod:**
```
Cost = 64 chips * 42.74 days * 24 hours/day * $2.50/chip-hour = $164,102
```

### Step 7: Summary comparison

| Metric | GPU Cluster (64x H100) | TPU Pod (64x v4) |
|---|---|---|
| Sustained throughput | 31,680 TFLOPS | 10,560 TFLOPS |
| Training time | 14.2 days | 42.7 days |
| Comm. overhead | ~1.6% | ~0.2% |
| Memory per device | 80 GB | 32 GB |
| Total memory | 5,120 GB | 2,048 GB |
| Cost | $76,800 | $164,100 |
| Cost per TFLOPS-hour | $0.10 | $0.11 |

### Key takeaways

1. The H100 GPU cluster wins on absolute training time (3x faster) and total cost (2.1x cheaper) due to the H100's significantly higher sustained throughput.

2. The TPU pod has a more uniform interconnect (all-to-all ICI vs hierarchical NVLink + InfiniBand), which becomes more important at larger scale when communication overhead is a bottleneck.

3. The cost per TFLOPS-hour is similar between platforms, suggesting that cloud pricing roughly tracks hardware performance.

4. The GPU cluster's larger memory per device (80 GB vs 32 GB) provides more flexibility for larger batch sizes and models without requiring aggressive memory optimization.

5. In practice, software ecosystem factors (PyTorch vs JAX, CUDA vs XLA, library availability) often dominate the hardware performance comparison in the decision-making process.
