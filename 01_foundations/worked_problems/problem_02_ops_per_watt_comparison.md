# Worked Problem: Ops per Watt Comparison

## Problem Statement

You are tasked with selecting an AI accelerator for a data center inference deployment. The data center has 10 MW of total power budget for AI inference (including cooling overhead of 1.3x PUE). You need to serve a transformer model that requires 2 * 10^12 INT8 operations per inference request at a target throughput of 50,000 requests per second.

Compare the following three accelerators:

| Accelerator | Peak INT8 TOPS | TDP (Watts) | Sustained utilization | Cost per chip |
|---|---|---|---|---|
| Chip A (GPU) | 1600 | 700 | 55% | $30,000 |
| Chip B (Custom ASIC) | 400 | 150 | 80% | $8,000 |
| Chip C (Edge-class) | 100 | 25 | 70% | $1,500 |

Determine:
1. The sustained TOPS for each chip.
2. The effective TOPS/W for each chip.
3. The number of chips needed to meet the throughput target.
4. The total power consumption (including PUE) for each solution.
5. Whether each solution fits within the 10 MW power budget.
6. The total chip cost for each solution.

---

## Worked Solution

### Step 1: Calculate sustained throughput per chip

```
Sustained TOPS = Peak TOPS * Utilization
```

- **Chip A**: 1600 * 0.55 = 880 TOPS
- **Chip B**: 400 * 0.80 = 320 TOPS
- **Chip C**: 100 * 0.70 = 70 TOPS

### Step 2: Calculate effective TOPS/W

```
TOPS/W = Sustained TOPS / TDP
```

- **Chip A**: 880 / 700 = 1.26 TOPS/W
- **Chip B**: 320 / 150 = 2.13 TOPS/W
- **Chip C**: 70 / 25 = 2.80 TOPS/W

Chip C has the best energy efficiency, followed by Chip B, then Chip A. This is typical: smaller, more specialized chips tend to achieve better TOPS/W because they have less overhead from unused hardware features.

### Step 3: Calculate required throughput

```
Total required TOPS = ops_per_request * requests_per_second
                    = 2 * 10^12 * 50,000
                    = 1.0 * 10^17 ops/second
                    = 100,000 TOPS
```

### Step 4: Calculate number of chips required

```
Num chips = Total required TOPS / Sustained TOPS per chip (rounded up)
```

- **Chip A**: 100,000 / 880 = 114 chips (rounded up)
- **Chip B**: 100,000 / 320 = 313 chips (rounded up)
- **Chip C**: 100,000 / 70 = 1,429 chips (rounded up)

### Step 5: Calculate total power consumption (including PUE)

```
Total power = Num chips * TDP * PUE
```

PUE (Power Usage Effectiveness) of 1.3 means for every watt of IT power, 0.3 watts are consumed by cooling and infrastructure.

- **Chip A**: 114 * 700 * 1.3 = 103,740 W = 103.7 kW
- **Chip B**: 313 * 150 * 1.3 = 61,035 W = 61.0 kW
- **Chip C**: 1,429 * 25 * 1.3 = 46,443 W = 46.4 kW

All three solutions fit well within the 10 MW budget. This is because the throughput requirement, while large, is modest relative to a 10 MW data center.

### Step 6: Calculate total chip cost

- **Chip A**: 114 * $30,000 = $3,420,000
- **Chip B**: 313 * $8,000 = $2,504,000
- **Chip C**: 1,429 * $1,500 = $2,143,500

### Step 7: Calculate 3-year TCO (simplified)

Assume electricity costs $0.08/kWh and chips are amortized over 3 years:

```
3-year electricity cost = Power (kW) * 8,760 hours/year * 3 years * $0.08/kWh
```

- **Chip A**: 103.7 * 8,760 * 3 * 0.08 = $218,092
- **Chip B**: 61.0 * 8,760 * 3 * 0.08 = $128,390
- **Chip C**: 46.4 * 8,760 * 3 * 0.08 = $97,658

**Total 3-year TCO (chips + electricity):**
- **Chip A**: $3,420,000 + $218,092 = $3,638,092
- **Chip B**: $2,504,000 + $128,390 = $2,632,390
- **Chip C**: $2,143,500 + $97,658 = $2,241,158

### Step 8: Analysis

| Metric | Chip A | Chip B | Chip C |
|---|---|---|---|
| Sustained TOPS/W | 1.26 | 2.13 | 2.80 |
| Chips needed | 114 | 313 | 1,429 |
| Total power (kW) | 103.7 | 61.0 | 46.4 |
| Chip cost | $3.42M | $2.50M | $2.14M |
| 3-year TCO | $3.64M | $2.63M | $2.24M |

Chip C has the lowest TCO due to superior energy efficiency and low per-unit cost. However, this analysis omits important practical considerations:

- **Rack space and networking**: 1,429 chips require significantly more rack space, network switches, and management infrastructure than 114 chips.
- **Latency**: Small chips with lower throughput per chip may have higher per-request latency if the model cannot be partitioned across them effectively.
- **Software ecosystem**: GPUs (Chip A) typically have more mature software stacks, reducing development time and risk.
- **Availability and supply**: Custom ASICs and edge chips may have longer lead times.

In practice, the optimal choice depends on the specific deployment constraints and requires a more detailed TCO model including networking, cooling infrastructure, software development, and operational costs.
