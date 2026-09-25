# Worked Problem: Chip-to-Chip Interconnect

## Problem Statement

You are designing a multi-chip AI training system with 8 accelerator chips. Each chip provides 500 TFLOPS of BF16 compute with 80 GB HBM at 2 TB/s. The system will train a 30B parameter model using tensor parallelism across all 8 chips.

The model has 60 layers, D=7168, FFN dimension=28672, 56 attention heads.

Determine:
1. The activation tensor size communicated per layer
2. The total inter-chip communication per forward+backward pass
3. The minimum inter-chip bandwidth to keep communication overhead below 10%
4. Whether NVLink (900 GB/s) or PCIe Gen5 (128 GB/s) meets the requirement

---

## Worked Solution

### Step 1: Activation tensor size per layer

With tensor parallelism, each linear layer (QKV projection, output projection, FFN up/down) is split across 8 chips along the output dimension. After each split linear layer, an all-reduce of the activation tensor is needed to combine partial results.

Activation tensor shape per layer for a batch: (B*S, D) where B*S is the total tokens in the micro-batch.

Assume micro-batch of B=4, S=2048: B*S = 8192 tokens.

```
Activation size = 8192 * 7168 * 2 bytes (BF16) = 117.4 MB
```

### Step 2: All-reduces per layer

Each transformer layer has 2 all-reduce operations for tensor parallelism:
1. After the attention output projection (combining split outputs)
2. After the FFN down projection

For the backward pass, there are 2 additional all-reduces (for activation gradients).

Total all-reduces per layer: 4 (2 forward + 2 backward)

### Step 3: Total communication per forward+backward pass

```
Total all-reduce calls = 60 layers * 4 = 240
Data per all-reduce = 117.4 MB (each node sends and receives this amount)
```

For a ring all-reduce with 8 chips:
```
Communication per all-reduce = 2 * (7/8) * 117.4 MB = 205.5 MB (per chip)
Total communication = 240 * 205.5 MB = 49.3 GB (per chip)
```

### Step 4: Compute time for forward+backward pass

Total compute per step (using 6ND approximation per micro-batch):
```
FLOPS = 6 * 30 * 10^9 * 8192 = 1.47 * 10^15 FLOPS per step
```

With 8 chips * 500 TFLOPS each at 50% utilization:
```
Compute time = 1.47 * 10^15 / (8 * 500 * 10^12 * 0.5) = 0.736 seconds
```

### Step 5: Required bandwidth for 10% overhead

For communication overhead < 10% of compute time:
```
Max communication time = 0.1 * 0.736 = 0.0736 seconds = 73.6 ms
Required bandwidth = 49.3 GB / 0.0736 s = 670 GB/s per chip
```

However, communication can overlap with computation. If 80% overlap is achieved:
```
Non-overlapped communication = 0.2 * 49.3 GB = 9.86 GB
Required bandwidth for non-overlapped portion = 9.86 / 0.0736 = 134 GB/s
```

### Step 6: Compare with NVLink and PCIe

The 49.3 GB is what each chip *sends* (it receives the same amount at the same time), so the relevant figure is the per-direction bandwidth: half of each bidirectional number.

**NVLink 4.0 (900 GB/s bidirectional = 450 GB/s per direction)**:
```
Communication time (no overlap) = 49.3 GB / 450 GB/s = 109.6 ms
Overhead = 109.6 / 736 = 14.9% (misses the 10% target without overlap; 80% overlap: 21.9 ms / 736 = 3.0%)
```

**PCIe Gen5 x16 (128 GB/s bidirectional = 64 GB/s per direction)**:
```
Communication time (no overlap) = 49.3 GB / 64 GB/s = 770 ms
Overhead = 770 / 736 = 105% (far exceeds 10%, even with 80% overlap: 154 ms / 736 = 20.9%)
```

### Step 7: Summary

| Interconnect | Comm time | Overhead (no overlap) | Overhead (80% overlap) |
|---|---|---|---|
| NVLink 900 GB/s | 109.6 ms | 14.9% | 3.0% |
| PCIe 128 GB/s | 770 ms | 105% | 20.9% |

**Conclusion**: NVLink meets the requirement once communication is overlapped with compute (the 670 GB/s no-overlap requirement exceeds its 450 GB/s per direction, but the 134 GB/s needed with 80% overlap is well within it). PCIe misses it even with aggressive overlap (80%), and in practice the overhead would be higher still due to non-overlappable communication at layer boundaries and the need for synchronization.

This explains why tensor parallelism is exclusively used within NVLink domains: the per-layer communication overhead is too high for lower-bandwidth interconnects.
