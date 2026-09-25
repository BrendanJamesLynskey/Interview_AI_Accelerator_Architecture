# Worked Problem: Dataflow Scheduling

## Problem Statement

You are designing a spatial dataflow accelerator with a 4x4 grid of Processing Elements (PEs) connected in a 2D mesh. Each PE has:
- One INT8 MAC unit (1 MAC per cycle)
- 1 KB of local SRAM (scratchpad)
- Connections to 4 neighbors (N, S, E, W) with 1-cycle latency per hop

You need to map a small GEMM: C = A * B where A is 4x8, B is 8x4, producing C (4x4). All elements are INT8, accumulation in INT32.

Design:
1. An output-stationary mapping that assigns one output element to each PE
2. The data movement schedule showing which data travels where each cycle
3. Calculate total cycles and utilization
4. Compare with a weight-stationary alternative

---

## Worked Solution

### Step 1: Output-stationary mapping

In an output-stationary dataflow, each PE computes one element of the output matrix C. Since C is 4x4 and we have a 4x4 PE grid, the mapping is direct:

```
PE(i,j) computes C[i][j] = sum over k: A[i][k] * B[k][j]   for k = 0..7
```

Each PE needs to receive:
- Row i of A: 8 elements (A[i][0] through A[i][7])
- Column j of B: 8 elements (B[0][j] through B[7][j])

Each PE performs 8 MACs to produce one output element.

### Step 2: Data delivery schedule

**A matrix delivery (rows):** Each row of A must reach all PEs in that row. We can broadcast each row along the west edge and have it propagate eastward.

```
Cycle 0: A[0][0] enters PE(0,0); A[1][0] enters PE(1,0); A[2][0] enters PE(2,0); A[3][0] enters PE(3,0)
Cycle 1: A[i][0] propagates to PE(i,1); A[i][1] enters PE(i,0)
Cycle 2: A[i][0] at PE(i,2); A[i][1] at PE(i,1); A[i][2] enters PE(i,0)
...
```

Each row element takes (column_index) extra cycles to reach PE(i, column_index) due to east-ward propagation. But since all PEs in a row need the same A element, multicast along the row is ideal.

**B matrix delivery (columns):** Each column of B must reach all PEs in that column. We can inject from the north edge and propagate southward.

```
Cycle 0: B[0][0] enters PE(0,0); B[0][1] enters PE(0,1); B[0][2] enters PE(0,2); B[0][3] enters PE(0,3)
Cycle 1: B[0][j] propagates to PE(1,j); B[1][j] enters PE(0,j)
...
```

### Step 3: Synchronized schedule

For PE(i,j) to perform a MAC at cycle t, it needs A[i][k] and B[k][j] simultaneously. We need to synchronize arrival of corresponding A and B elements.

Using skewed input delivery (similar to systolic array):
- A elements enter from the west edge, row i starts at cycle i (row skew)
- B elements enter from the north edge, column j starts at cycle j (column skew)

Schedule for PE(i,j) receiving the k-th pair:
```
A[i][k] arrives at PE(i,j) at cycle: i + k + j  (i skew + k injection time + j propagation hops)
B[k][j] arrives at PE(i,j) at cycle: j + k + i  (j skew + k injection time + i propagation hops)
```

These are equal, so the skewed delivery naturally synchronizes A and B elements.

PE(i,j) performs MACs at cycles: i + j, i + j + 1, ..., i + j + 7 (for k = 0 to 7).

### Step 4: Calculate total cycles

- First MAC: PE(0,0) at cycle 0
- Last MAC: PE(3,3) at cycle 3 + 3 + 7 = 13

Total cycles from first to last result: 14 cycles (cycles 0 through 13)

Including output drain (writing C elements, which can start as soon as each PE finishes):
- PE(0,0) finishes at cycle 7
- PE(3,3) finishes at cycle 13
- Output drain: 1 cycle per PE (can overlap with computation of later PEs)
- Total: approximately 14 cycles

### Step 5: Calculate utilization

**Useful MACs:** 4 * 8 * 4 = 128 MACs (M * K * N)

**Total PE-cycles available:** 16 PEs * 14 cycles = 224 PE-cycles

**MAC utilization:**
```
Utilization = 128 / 224 = 57.1%
```

The loss comes from the skewed startup: PE(0,0) is active for 8 of 14 cycles, while PE(3,3) is active for 8 of 14 cycles, but they are active during different time windows. The fill and drain phases waste PE-cycles.

### Step 6: Weight-stationary alternative

In a weight-stationary approach, we preload weights into PEs and stream activations through.

Since B is 8x4, we cannot fit all 32 weight elements into 16 PEs (each PE would need 2 weights). Instead, we process in two phases:

**Phase 1 (k = 0..3):** Load B[0:4][0:4] into the 4x4 PE grid. PE(k,j) holds B[k][j].
- Stream rows of A[i][0:4] from the west edge (4 rows, 4 elements each)
- Each row flows through all 4 PEs in each column, accumulating partial sums
- Partial sums for C[i][j] accumulate as data flows south through column j

**Phase 2 (k = 4..7):** Load B[4:8][0:4] into the grid. Stream A[i][4:8].
- Accumulate into the same partial sums from Phase 1.

**Cycles per phase:**
- Weight load: 4 cycles (loading 4 weights per column, sequentially)
- Compute: 4 (rows of A) + 4 (eastward skew across the 4 columns) + 4 (southward propagation through each column) - 2 = 10 cycles (the same $M + N + K - 2$ skew formula that gives 14 cycles for the output-stationary case)
- Phase total: 4 + 10 = 14 cycles

**Total cycles:** 2 phases * 14 cycles = 28 cycles

**Utilization:** 128 MACs / (16 PEs * 28 cycles) = 128 / 448 = 28.6%

### Step 7: Comparison

| Metric | Output-Stationary | Weight-Stationary |
|---|---|---|
| Total cycles | 14 | 28 |
| Utilization | 57.1% | 28.6% |
| Data movement (A) | 4 rows * 8 elements = 32 | 4 rows * 8 elements = 32 |
| Data movement (B) | 4 cols * 8 elements = 32 | 2 loads * 16 weights = 32 |
| Partial sum movement | None (stays in PE) | Through column (internal) |
| Weight reload | None | Once (2 phases) |

For this small problem, output-stationary is more efficient because:
1. No weight reloading overhead
2. The 4x4 output maps perfectly to the 4x4 PE grid
3. Partial sums never leave the PE

However, weight-stationary would be more efficient for larger batch sizes (many A matrices with the same B), because the weight load cost is amortized across all inputs.

This illustrates the fundamental tradeoff in dataflow selection: the optimal choice depends on the specific matrix dimensions and the ratio of reuse opportunities for each data type.
