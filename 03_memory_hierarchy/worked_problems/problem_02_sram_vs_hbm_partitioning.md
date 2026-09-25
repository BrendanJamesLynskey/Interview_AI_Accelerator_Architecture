# Worked Problem: SRAM vs HBM Partitioning

## Problem Statement

You are designing an AI accelerator with a fixed die area budget. You must allocate area between:
- MAC array compute units (each 16x16 INT8 MAC array = 0.1 mm^2, providing 512 INT8 ops/cycle at 1 GHz = 0.512 TOPS)
- On-chip SRAM (density: 25 Mbit/mm^2 = 3.125 MB/mm^2)

The chip has:
- Total die area available for compute + SRAM: 400 mm^2 (remaining area used for I/O, control, interconnect)
- HBM bandwidth: 2 TB/s (fixed, not part of the area budget)
- Clock frequency: 1 GHz

The target workload is tiled GEMM with dimensions M=N=K=4096, INT8.

Analyze three design points:
- Design A: 75% compute, 25% SRAM
- Design B: 50% compute, 50% SRAM
- Design C: 25% compute, 75% SRAM

For each design, determine the achievable GEMM performance.

---

## Worked Solution

### Step 1: Calculate resources for each design

| Resource | Design A | Design B | Design C |
|---|---|---|---|
| Compute area | 300 mm^2 | 200 mm^2 | 100 mm^2 |
| SRAM area | 100 mm^2 | 200 mm^2 | 300 mm^2 |
| MAC arrays | 3000 | 2000 | 1000 |
| Peak INT8 TOPS | 1536 | 1024 | 512 |
| SRAM capacity | 312.5 MB | 625 MB | 937.5 MB |

### Step 2: Determine optimal tile sizes

For a tiled GEMM C = A * B with tiles (Tm, Tn, Tk), the SRAM must hold:
- A tile: Tm * Tk bytes (INT8)
- B tile: Tk * Tn bytes (INT8)
- C tile: Tm * Tn * 4 bytes (INT32 accumulation)

With double buffering for A and B tiles:
```
SRAM = 2 * (Tm * Tk + Tk * Tn) + Tm * Tn * 4
```

For square tiles (Tm = Tn = T) with Tk = T:
```
SRAM = 2 * 2 * T^2 + 4 * T^2 = 8 * T^2 bytes
```

Solving for T:
- **Design A** (312.5 MB): T = sqrt(312.5 * 10^6 / 8) = sqrt(39.06 * 10^6) = 6250 -> T = 6250. Since M=N=K=4096, T = min(6250, 4096) = 4096. The entire matrix fits!
- **Design B** (625 MB): T = sqrt(78.1 * 10^6) = 8838 -> T = 4096. Also fits entirely.
- **Design C** (937.5 MB): T = 4096. Also fits.

Wait -- all three designs have enough SRAM to hold the entire 4096x4096 matrices! Let us verify:
```
A (INT8): 4096 * 4096 = 16.8 MB
B (INT8): 4096 * 4096 = 16.8 MB
C (INT32): 4096 * 4096 * 4 = 67.1 MB
Total: 100.7 MB
With double buffering for A, B: 2 * 33.6 + 67.1 = 134.2 MB
```

Yes, even Design A (312.5 MB) can hold all data on-chip. This means HBM is accessed only once to load A and B and once to store C, and the GEMM is entirely compute-bound.

Let us use a larger problem to make the analysis interesting. Consider M=N=K=16384.

### Step 3: Recalculate for M=N=K=16384

Data sizes:
```
A (INT8): 16384^2 = 268 MB
B (INT8): 16384^2 = 268 MB  
C (INT32): 16384^2 * 4 = 1074 MB
Total: 1611 MB (too large for any design's SRAM)
```

Now tiling is necessary. With double-buffered A and B, and C staying on-chip for one output tile:
```
SRAM = 2 * (Tm * Tk + Tk * Tn) + Tm * Tn * 4
```

Setting Tk = 1024 (a practical choice) and Tm = Tn = T:
```
SRAM = 2 * 2 * T * 1024 + 4 * T^2 = 4096T + 4T^2
```

For each design, find maximum T:
- **Design A** (312.5 MB = 312.5 * 10^6): 4T^2 + 4096T = 312.5 * 10^6 -> T ≈ 8340. Use T = 8192.
- **Design B** (625 MB): T ≈ 12000. Use T = 8192.
- **Design C** (937.5 MB): T ≈ 14800. Use T = 8192.

All can fit T = 8192 with Tk = 1024. Actually, let us try more granular analysis.

With T = 8192 and Tk = 1024:
```
SRAM = 2 * (8192*1024 + 1024*8192) + 8192*8192*4 = 2*16.78M + 268.4M = 301.99 MB
```

Design A (312.5 MB): Fits. Number of output tiles = (16384/8192)^2 = 4.
Design B and C: Also fit with room to spare.

The number of K-tiles = 16384 / 1024 = 16. Total tile computations = 4 * 16 = 64.

### Step 4: Calculate HBM data movement

For each output tile (there are 4), across 16 K-tiles:
```
A tiles loaded: 16 * (8192 * 1024) bytes = 16 * 8.39 MB = 134.2 MB per output tile
B tiles loaded: 16 * (1024 * 8192) bytes = 134.2 MB per output tile
C written: 8192 * 8192 * 4 = 268.4 MB per output tile
```

Total HBM traffic: 4 * (134.2 + 134.2 + 268.4) = 4 * 536.9 = 2147 MB = 2.15 GB

(Loading A and B only once each — 268.4 + 268.4 + 1073.7 = 1611 MB in total — would need a schedule that keeps a whole row-block of A or column-block of B on chip alongside the C tile, which does not fit here. Each A and B block is therefore loaded twice, once per output tile that uses it.)

### Step 5: Calculate execution time for each design

Total compute: 2 * 16384^3 = 8.80 * 10^12 INT8 ops

**Compute time:**
- Design A: 8.80 * 10^12 / (1536 * 10^12) = 5.73 ms
- Design B: 8.80 * 10^12 / (1024 * 10^12) = 8.59 ms
- Design C: 8.80 * 10^12 / (512 * 10^12) = 17.19 ms

**HBM transfer time** (2147 MB at 2 TB/s):
- 2147 MB / 2000 GB/s = 1.07 ms

With double buffering, HBM transfer overlaps with compute. Since compute time >> HBM time for all designs, the workload is compute-bound.

### Step 6: Effective performance

| Metric | Design A | Design B | Design C |
|---|---|---|---|
| Peak TOPS | 1536 | 1024 | 512 |
| SRAM | 312.5 MB | 625 MB | 937.5 MB |
| Compute time | 5.73 ms | 8.59 ms | 17.19 ms |
| HBM time | 1.07 ms | 1.07 ms | 1.07 ms |
| Bottleneck | Compute | Compute | Compute |
| Effective TOPS | ~1536 | ~1024 | ~512 |

### Step 7: Analysis

For this large, compute-bound GEMM (16384^3), Design A wins decisively: it provides 3x more throughput than Design C because the GEMM is entirely compute-bound and the extra SRAM in Design C is unused.

However, the analysis changes for smaller or memory-bound operations:
- For batch-1 inference (matrix-vector multiplies), the arithmetic intensity is ~1 FLOP/byte. All designs would be HBM-bandwidth-bound at 2 TB/s, and the extra compute in Design A would be wasted.
- For medium-sized GEMMs where tile size is constrained by SRAM, Design B or C could achieve higher arithmetic intensity through larger tiles.

**Conclusion**: The optimal compute-to-SRAM ratio depends on the target workload mix. For training (large-batch, compute-bound GEMMs), prioritize compute (Design A). For inference (small-batch, memory-bound), a balanced design (B) or SRAM-heavy design (C) may be better. In practice, most commercial accelerators target Design A or B, because the dominant use case (training large models) is compute-bound.
