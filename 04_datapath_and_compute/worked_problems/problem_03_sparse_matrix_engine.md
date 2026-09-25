# Worked Problem: Sparse Matrix Engine

## Problem Statement

Design a hardware unit that performs sparse-dense matrix multiplication using 2:4 structured sparsity. The sparse matrix A (2:4 format) is multiplied with dense matrix B.

Specifications:
- Tile size: 16x16 output tile computed per cycle group
- Inner dimension K processed in chunks of 32
- Data type: FP16 inputs, FP32 accumulation
- Target: process one 16x16x32 tile every 16 cycles

Calculate:
1. The number of MAC units required
2. The metadata decode logic
3. The multiplexer network for operand selection
4. Area and power estimates compared to a dense-only design

---

## Worked Solution

### Step 1: Understand the 2:4 sparse tile computation

For a 16x16 output tile with K=32 inner dimension:
- Dense computation: 16 * 16 * 32 = 8192 MACs
- With 2:4 sparsity on A: only 2 out of every 4 K-elements are nonzero, so effective MACs = 16 * 16 * 32 * (2/4) = 4096 MACs

To complete in 16 cycles: 4096 / 16 = 256 MACs per cycle.

### Step 2: MAC array design

With 256 MACs per cycle, we need 256 FP16 MAC units (with FP32 accumulators).

Organization: A 16x16 array of MAC units, where each MAC unit computes one output element of the 16x16 output tile. Each MAC unit performs 256/16 = 16 MACs over 16 cycles (processing K=32 with 2:4 sparsity: 32 * 2/4 = 16 nonzero multiplies per output element).

Each cycle, each MAC unit processes one pair of nonzero values: 1 MAC/cycle * 16 cycles = 16 MACs per output element.

### Step 3: Metadata decode logic

For each row of A (16 rows), the K=32 elements are divided into 32/4 = 8 groups of 4. Each group has a 4-bit metadata encoding which 2 of 4 positions are nonzero.

Over the 16 cycles each row consumes its 8 groups (16 nonzeros), and each MAC performs 1 MAC per cycle, so per cycle each row consumes one nonzero — half a group — together with its 2-bit position index.

Metadata per row per cycle: 2 bits (one group's 4 bits every 2 cycles)
Total metadata decode: 16 rows * 2 bits = 32 bits per cycle

The decoder converts each 4-bit metadata into two 2-bit select signals:
```
Metadata 4 bits -> Select_0 (2 bits: position 0-3) + Select_1 (2 bits: position 0-3)
```

There are C(4,2) = 6 valid patterns. A small lookup table (6 entries) per group suffices.

Total decoders: one per row (each decodes a new group every 2 cycles) = 16 decoders.
Each decoder: ~20 gates. Total: ~320 gates (negligible area).

### Step 4: Multiplexer network

For each MAC unit, we need to select 2 elements from B matrix (one per nonzero in the group).

Each group of 4 elements in B requires two 4:1 multiplexers (one per nonzero):
```
Mux_0: selects B[k+select_0] from {B[k], B[k+1], B[k+2], B[k+3]}
Mux_1: selects B[k+select_1] from {B[k], B[k+1], B[k+2], B[k+3]}
```

Each 4:1 FP16 mux: ~16 bits * 3 gates per bit (for a 2-stage mux) = ~48 gates
Per MAC unit: 2 muxes (for 2 groups per cycle) * 2 (nonzero selections) = 4 muxes
Wait -- more precisely, per cycle we process 2 groups, so we need 2 * 2 = 4 selections per row, but each MAC only needs 2 (since it processes 2 nonzero values per cycle for accumulation).

Let us simplify: per MAC unit per cycle, we need 1 A value (already selected from compressed storage) and 1 B value (selected via mux). Over 16 cycles, each MAC accumulates 16 products.

Per MAC: one 4:1 FP16 multiplexer = ~48 gates.
Total: 256 MACs * 48 gates = 12,288 gates.
At 7nm (~2 gates/um^2 including routing): ~6,144 um^2 = 0.006 mm^2. Negligible.

### Step 5: Area comparison

**Dense-only design (same throughput):**

For the same 16x16x32 tile in 16 cycles without sparsity:
- MACs per cycle = 16*16*32/16 = 512 MACs
- 512 FP16 MAC units needed

FP16 MAC area: ~350 um^2 each
Dense array: 512 * 350 = 179,200 um^2 = 0.179 mm^2

**Sparse design:**
- 256 FP16 MAC units: 256 * 350 = 89,600 um^2
- Metadata decode: ~320 gates ≈ negligible
- Multiplexers: ~12,288 gates ≈ 0.006 mm^2
- Metadata storage: 32 bits/cycle * 16 cycles = 512 bits = 64 bytes buffer ≈ negligible
- Total: ~0.096 mm^2

**Comparison:**

| Design | MAC units | Total area | Effective throughput |
|---|---|---|---|
| Dense | 512 | 0.179 mm^2 | 8192 MACs / 16 cycles |
| Sparse (2:4) | 256 | 0.096 mm^2 | 4096 MACs / 16 cycles (= 8192 effective) |

The sparse design achieves the same effective throughput (in terms of equivalent dense operations) at ~54% of the area. The overhead for sparsity support (decoders + muxes) is about 7% of the MAC area (0.006 of 0.090 mm^2).

### Step 6: Power comparison

FP16 MAC energy: ~0.4 pJ per operation.

Dense: 512 MACs * 0.4 pJ * 1 GHz = 204.8 mW
Sparse: 256 MACs * 0.4 pJ * 1 GHz = 102.4 mW + ~2 mW (mux + decode) = 104.4 mW

**The sparse design uses ~51% of the power for the same effective throughput**, because it performs half the multiplications (the zero-valued ones are skipped entirely rather than computed and discarded).

### Conclusion

The 2:4 structured sparsity hardware support adds modest area overhead (~7%) while enabling 2x effective throughput at nearly 2x better energy efficiency compared to a dense design with the same silicon area. This explains why NVIDIA included this feature in the A100 and subsequent architectures -- the hardware cost is negligible but the benefit is substantial for compatible workloads.
