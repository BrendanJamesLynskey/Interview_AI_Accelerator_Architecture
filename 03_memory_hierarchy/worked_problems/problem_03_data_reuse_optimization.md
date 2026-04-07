# Worked Problem: Data Reuse Optimization

## Problem Statement

You need to compute a 2D convolution on an accelerator with 512 KB of on-chip SRAM. The convolution parameters are:

- Input: 1 image, 64 channels, 56x56 spatial (NCHW)
- Filter: 128 output channels, 64 input channels, 3x3 kernel
- Output: 1 image, 128 channels, 56x56 spatial (with padding=1, stride=1)
- Data type: FP16 (2 bytes per element)

Calculate:
1. The total data sizes (input, weights, output)
2. Whether all data fits on-chip
3. A tiling strategy that maximizes data reuse within the 512 KB SRAM
4. The total HBM traffic with and without tiling
5. The arithmetic intensity achieved

---

## Worked Solution

### Step 1: Calculate total data sizes

**Input activations:**
```
Size = 1 * 64 * 56 * 56 * 2 = 401,408 bytes = 392 KB
```

**Weights:**
```
Size = 128 * 64 * 3 * 3 * 2 = 147,456 bytes = 144 KB
```

**Output activations:**
```
Size = 1 * 128 * 56 * 56 * 2 = 802,816 bytes = 784 KB
```

**Total: 392 + 144 + 784 = 1320 KB**

### Step 2: Check if all data fits on-chip

Available SRAM: 512 KB. Total data: 1320 KB. Does not fit.

Even just input + weights = 536 KB, which slightly exceeds 512 KB. We must tile.

### Step 3: Design a tiling strategy

We will tile across output channels (C_out) and spatial dimensions (H, W). Input channels (C_in) will be processed as the reduction dimension.

**Tile parameters:**
- Tc_out: number of output channels per tile
- Th, Tw: spatial tile dimensions for the output
- Tc_in: process all 64 input channels (reduction dimension, not tiled, but we may need to tile if weights are too large)

**SRAM allocation per tile:**
- Input tile: Tc_in * (Th+2) * (Tw+2) * 2 bytes (need +2 for 3x3 halo with padding=1)
- Weight tile: Tc_out * Tc_in * 3 * 3 * 2 bytes
- Output tile: Tc_out * Th * Tw * 4 bytes (FP32 accumulation until final write)

Let us try Tc_out = 32, Th = Tw = 28 (splitting 56x56 into 2x2 spatial tiles), Tc_in = 64:

```
Input tile: 64 * 30 * 30 * 2 = 115,200 bytes = 112.5 KB
Weight tile: 32 * 64 * 9 * 2 = 36,864 bytes = 36 KB
Output tile: 32 * 28 * 28 * 4 = 100,352 bytes = 98 KB
Total: 112.5 + 36 + 98 = 246.5 KB
```

This fits in 512 KB with room for double buffering. With double buffering of input and weight tiles:
```
Total with double buffer: 2 * (112.5 + 36) + 98 = 297 + 98 = 395 KB
```

Still fits in 512 KB.

### Step 4: Count the tiles and data movement

**Output tiles**: (128/32) * (56/28) * (56/28) = 4 * 2 * 2 = 16 tiles

For each output tile:
- **Input tile load**: 64 * 30 * 30 * 2 = 115,200 bytes
  - The same input tile is reused across different Tc_out tiles at the same spatial position.
  - Spatial positions sharing: 4 output-channel tiles share the same spatial region.
  - Load count: 2 * 2 = 4 spatial tile positions (each loaded 4 times for different output channel tiles, BUT we can reorder to reuse)

**Optimal loop ordering to maximize reuse:**

```
for th in range(0, 56, 28):       # 2 height tiles
  for tw in range(0, 56, 28):     # 2 width tiles
    Load input tile [64, th:th+30, tw:tw+30]  -- 112.5 KB
    for tc_out in range(0, 128, 32):  # 4 output channel tiles
      Load weight tile [tc_out:tc_out+32, 64, 3, 3]  -- 36 KB
      Compute output tile [tc_out:tc_out+32, th:th+28, tw:tw+28]
      Store output tile -- 98 KB
```

With this ordering, each input tile is loaded once and reused across 4 output channel tiles.

**Total HBM traffic:**

Input loads: 4 spatial tiles * 112.5 KB = 450 KB

Weight loads: 4 spatial tiles * 4 output-channel tiles * 36 KB = 576 KB
(Weights are reloaded for each spatial tile. To avoid this, we could reorder loops.)

Alternative loop ordering (weight-reuse first):
```
for tc_out in range(0, 128, 32):  # 4 output channel tiles
  Load weight tile once
  for th, tw spatial tiles:
    Load input tile
    Compute, store output
```

Weight loads: 4 * 36 KB = 144 KB (each weight tile loaded once)
Input loads: 4 output-channel groups * 4 spatial tiles * 112.5 KB = 1800 KB
(Input is reloaded for each output channel group)

Neither ordering is ideal. The best is to keep both in SRAM when possible. Since weight tile (36 KB) + input tile (112.5 KB) = 148.5 KB, and we have 512 KB, we could keep ALL weight tiles in SRAM:

Total weights: 144 KB < 512 KB. So load all weights once, then stream spatial input tiles:

```
Load ALL weights: 144 KB (stays in SRAM)
for th, tw spatial tiles (4 tiles):
  Load input tile: 112.5 KB
  for tc_out (4 groups, processing from weights already in SRAM):
    Compute output tile
    Store output tile: 98 KB
```

SRAM usage: 144 KB (all weights) + 112.5 KB (input tile) + 98 KB (output tile) = 354.5 KB. Fits!

### Step 5: Calculate optimized HBM traffic

```
Weight loads: 144 KB (loaded once)
Input loads: 4 * 112.5 KB = 450 KB (each spatial tile loaded once)
Output stores: 16 tiles * 98 KB = 1568 KB (simplified: 784 KB actual since output is FP16)
   Actually, output is accumulated in FP32 on-chip then written as FP16: 16 * 32 * 28 * 28 * 2 = 784 KB
Total HBM traffic: 144 + 450 + 784 = 1378 KB = 1.35 MB
```

### Step 6: Calculate traffic WITHOUT tiling (naive case)

Without tiling, every multiply requires loading operands from HBM:
- Each output element (128 * 56 * 56 = 401,408 elements) requires 64 * 9 = 576 multiply-adds
- Each multiply loads one weight and one activation from HBM
- Total reads: 401,408 * 576 * 2 * 2 = ~924 MB (extremely wasteful)

A more realistic "no tiling" baseline is loading each matrix once per output element computation:
- Input (392 KB) + Weights (144 KB) loaded per spatial output position = clearly impractical

The fair comparison is simply: with tiling, total traffic = 1.35 MB. Without any reuse optimization (each input loaded per output channel group): traffic = 4 * 392 + 144 + 784 = 2.5 MB.

**Tiling saves approximately 46% of HBM traffic in this case.**

### Step 7: Calculate arithmetic intensity

**Total FLOPS:**
```
FLOPS = 2 * 128 * 64 * 3 * 3 * 56 * 56 = 2 * 128 * 64 * 9 * 3136 = 462,422,016 = 462.4 MFLOPS
```

**Arithmetic intensity (with tiling):**
```
AI = 462.4 * 10^6 / (1.35 * 10^6) = 342.5 FLOPS/byte
```

This is above the ridge point of most modern accelerators (150-295 FLOPS/byte), confirming that this tiled convolution is compute-bound -- a good result.

Without tiling optimization (2.5 MB traffic):
```
AI = 462.4 / 2.5 = 184.9 FLOPS/byte
```

Still likely compute-bound on most hardware, but with less margin. The tiling optimization provides insurance against being memory-bound and reduces HBM energy consumption.
