# Worked Problem: Systolic Array Mapping

## Problem Statement

You have a 128 x 128 weight-stationary systolic array running at 1 GHz. Each PE performs one INT8 multiply-accumulate per cycle with FP32 accumulation.

Map the following GEMM operation onto this array:
- Matrix A: 512 x 1024 (activations)
- Matrix B: 1024 x 768 (weights)
- Output C: 512 x 768

Calculate:
1. The number of output tiles
2. The number of weight loads
3. The total number of cycles (including fill/drain overhead)
4. The achieved throughput in TOPS
5. The utilization compared to peak

---

## Worked Solution

### Step 1: Determine the tiling strategy

The systolic array is 128 x 128. We tile the matrices as follows:
- M dimension (512): ceil(512/128) = 4 tiles
- N dimension (768): ceil(768/128) = 6 tiles
- K dimension (1024): ceil(1024/128) = 8 inner tiles

Total output tiles: 4 * 6 = 24 tiles, each 128 x 128.

Note: For the N dimension, 768 / 128 = 6.0 (exact), so no PE waste. For the M dimension, 512 / 128 = 4.0 (exact). Good alignment.

### Step 2: Execution flow for weight-stationary dataflow

For each output tile (m, n):
1. Load the weight tile B[k, n] into the systolic array (128 x 128 weights)
2. Stream the activation tile A[m, k] through the array from left to right
3. Accumulate partial sums
4. Repeat for all K tiles (k = 0 to 7), accumulating into the same output tile
5. Write the completed output tile C[m, n]

For each output tile, we process K/128 = 8 inner tiles. Each inner tile involves streaming 128 rows of 128 activations through the array.

### Step 3: Calculate cycles per inner tile

For one inner tile (128x128 weight tile applied to 128x128 activation tile):
- Fill phase: 128 + 128 - 2 = 254 cycles (data propagating to the bottom-right PE)
- Steady state: 0 additional cycles (for a single 128-wide column of activations, the fill IS the computation)

More precisely, for a weight-stationary array computing a 128x128 output tile with K=128:
- 128 activation columns are streamed through the array
- Each column takes 1 cycle to enter (one element per PE per cycle along the left edge)
- The last result emerges 128 + 128 - 2 = 254 cycles after the first activation enters
- But with pipelining, a new activation column can enter every cycle

Total cycles for one inner tile (128 activations streamed through 128x128 array):
```
Cycles = 128 (streaming activations) + 127 (drain time for last activation to propagate) 
       = 255 cycles
```

Actually, let us be more precise. In a weight-stationary 128x128 systolic array:
- Row i of activations enters with a skew of i cycles (staggered input)
- The first result at PE(0,0) appears at cycle 0 (after 0 propagation delay for the corner PE)
- The last result at PE(127,127) appears at cycle 127 + 127 = 254 (skew of row 127 + propagation delay to column 127)

For K=128 elements streamed per row:
- First element of row 0 enters at cycle 0
- Last element of row 0 enters at cycle 127
- First element of row 127 enters at cycle 127 (due to row skew)
- Last element of row 127 enters at cycle 127 + 127 = 254
- Last result emerges at cycle 254 + 127 = 381? 

Let us simplify with the standard formula:
```
Cycles per (M_tile, N_tile, K_tile) = M_tile + N_tile + K_tile - 2
```

For M_tile = N_tile = K_tile = 128:
```
Cycles = 128 + 128 + 128 - 2 = 382 cycles
```

### Step 4: Account for pipelining across K tiles

For a single output tile processed across 8 K-tiles:
- If done sequentially: 8 * 382 = 3056 cycles
- If pipelined (loading next weight tile while draining current): Weight loading time dominates if we must change weights for each K tile

In weight-stationary mode, we keep weights fixed and stream different activation tiles. But here we need to process across the K dimension, which means for each (m, n) output tile:
- Load weights B[0:128, n_start:n_start+128] -- takes 128 cycles (128 rows of 128 weights)
- Stream A[m_start:m_start+128, 0:128] -- takes 382 cycles
- Load weights B[128:256, n_start:n_start+128] -- takes 128 cycles  
- Stream A[m_start:m_start+128, 128:256] -- takes 382 cycles
- ... repeat for all 8 K-tiles

Wait -- in weight-stationary, we change weights for each K tile. Total per output tile:
```
Cycles per output tile = 8 * (128 + 382) = 8 * 510 = 4080 cycles
```

With double-buffering (loading next weight tile while computing current):
```
Cycles per output tile = 128 + 8 * 382 = 128 + 3056 = 3184 cycles
```

(First weight load of 128 cycles, then 8 compute phases overlapped with weight loads.)

### Step 5: Total cycles for all output tiles

```
Total output tiles = 24
Total cycles = 24 * 3184 = 76,416 cycles
```

### Step 6: Calculate throughput

Total useful operations:
```
Operations = 2 * M * K * N = 2 * 512 * 1024 * 768 = 805,306,368 ops = 0.805 * 10^9
```

Time at 1 GHz:
```
Time = 76,416 / 10^9 = 76.4 microseconds
```

Throughput:
```
Throughput = 0.805 * 10^9 / 76.4 * 10^-6 = 10.54 TOPS
```

### Step 7: Calculate utilization

Peak throughput = 128 * 128 * 2 * 1 GHz = 32,768 * 10^9 = 32.77 TOPS (counting each MAC as 2 ops)

```
Utilization = 10.54 / 32.77 = 32.2%
```

### Step 8: Identify the utilization loss

The 32.2% utilization comes from:
1. **Fill/drain overhead**: Each 382-cycle compute phase has 254 cycles of fill/drain where not all PEs are active, vs 128 cycles of steady state. Compute efficiency = 128/382 = 33.5%.
2. **Weight loading overhead**: Even with double-buffering, the initial weight load adds cycles.

To improve utilization:
- **Increase K-tile size**: Using K_tile = 1024 (full K dimension) would give cycles = 128 + 128 + 1024 - 2 = 1278, with steady-state efficiency = 1024/1278 = 80.1%. However, this requires 128*1024 = 131,072 weight values in the array, which may exceed local PE storage.
- **Use larger matrices**: Bigger M, N, K dimensions amortize the fill/drain overhead.
- **Pipeline output tiles**: Begin loading weights for the next output tile while draining the current one.
