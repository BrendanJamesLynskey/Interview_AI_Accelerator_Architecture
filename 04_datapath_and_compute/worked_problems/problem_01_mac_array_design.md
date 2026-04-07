# Worked Problem: MAC Array Design

## Problem Statement

Design a MAC array for an AI inference accelerator with the following requirements:
- Target throughput: 128 TOPS INT8
- Clock frequency: 1 GHz
- Support INT8 multiply with INT32 accumulation
- Determine the number of MAC units, array organization, area estimate, and power estimate

---

## Worked Solution

### Step 1: Calculate required MAC units

Each INT8 MAC performs one multiply-accumulate per cycle = 2 INT8 operations (1 multiply + 1 add).

```
Required MACs = 128 TOPS / (2 ops/MAC/cycle * 1 GHz)
              = 128 * 10^12 / (2 * 10^9)
              = 64,000 MAC units
```

### Step 2: Choose array organization

A 256 x 250 array provides 64,000 MACs, but non-power-of-2 dimensions are inconvenient. Options:
- **256 x 256** = 65,536 MACs -> 131 TOPS (2.4% over target, acceptable)
- **4 arrays of 128 x 128** = 4 * 16,384 = 65,536 MACs

Choose: 4 independent 128x128 systolic arrays. This provides:
- Better utilization for smaller matrix dimensions (can use 1-4 arrays depending on workload)
- Independent scheduling of different operations on different arrays
- Manageable intra-array communication distances

### Step 3: Estimate area

At 7nm technology, an INT8 MAC unit (8x8 multiplier + 32-bit accumulator + control):
```
Area per MAC ≈ 105 um^2 (multiplier) + 50 um^2 (accumulator + control) = 155 um^2
```

Total MAC area:
```
65,536 * 155 um^2 = 10.16 mm^2
```

Including interconnect, local registers, and control (typically 2-3x the raw MAC area):
```
Total array area ≈ 10.16 * 2.5 = 25.4 mm^2
```

### Step 4: Estimate power

Energy per INT8 MAC operation at 7nm: approximately 0.1 pJ for the multiply, 0.05 pJ for the accumulate, 0.15 pJ for register read/write = 0.3 pJ total.

```
MAC dynamic power = 65,536 MACs * 0.3 pJ * 1 GHz = 19.66 mW per GHz = 19.66 W
```

Including clock distribution, control logic, and interconnect (typically 3-5x MAC power):
```
Total array power ≈ 19.66 * 4 = 78.6 W
```

### Step 5: Verify design metrics

| Metric | Value |
|---|---|
| MAC units | 65,536 (4 x 128x128 arrays) |
| Peak INT8 TOPS | 131 TOPS |
| Array area | ~25 mm^2 |
| Array power | ~79 W |
| TOPS/mm^2 | 5.2 |
| TOPS/W | 1.66 |

### Step 6: Context and comparison

For reference, the NVIDIA A100's tensor cores deliver approximately 624 TOPS INT8 on a 826 mm^2 die at 400W TDP (though the tensor cores are only a fraction of the die area and power).

Our design's 131 TOPS in 25 mm^2 at 79W would leave substantial die area for on-chip SRAM, HBM controllers, NoC, and other logic in a full accelerator design. The TOPS/W of 1.66 is reasonable for a dedicated inference engine -- comparable to edge AI accelerators, though the absolute throughput is data-center-class.

The key design choices that would follow are: (a) how much SRAM to pair with these MAC arrays (see [SRAM vs HBM partitioning](../../03_memory_hierarchy/worked_problems/problem_02_sram_vs_hbm_partitioning.md)), (b) the interconnect between arrays and memory, and (c) the control architecture (statically scheduled vs dynamically scheduled).
