# Worked Problem: Power Envelope Design

## Problem Statement

You are designing an AI inference accelerator for data center deployment with a 300W TDP target. The chip must deliver at least 400 TOPS INT8.

Allocate the power budget across:
- INT8 MAC array
- On-chip SRAM (target 64 MB)
- HBM3 interface (2 stacks, target 1.6 TB/s)
- NoC and control logic
- I/O (PCIe Gen5 x16)
- Clock distribution

Determine if the design is feasible at 5nm technology.

---

## Worked Solution

### Step 1: MAC array power

Target: 400 TOPS INT8 = 200T MAC ops/second (each MAC = 2 ops)

Energy per INT8 MAC at 5nm: approximately 0.08 pJ (slightly better than 7nm due to process improvement)

```
MAC dynamic power = 200 * 10^12 * 0.08 * 10^-12 = 16 W
```

This seems low. Let us include register file and local interconnect:
- Register read/write per MAC: ~0.2 pJ
- Total per MAC: 0.08 + 0.2 = 0.28 pJ

```
MAC + register power = 200T * 0.28 pJ = 56 W
```

### Step 2: SRAM power

64 MB SRAM at 5nm. Assume aggregate read bandwidth of 8 TB/s (to feed the MAC array):

Energy per SRAM read: ~3 pJ per 16-bit read (at 5nm with moderate wire length)
```
SRAM dynamic power = 8 * 10^12 bytes/s * 3 * 10^-12 J/byte = 24 W
SRAM leakage (64 MB at ~0.1 mW/MB at 5nm): 64 * 0.1 = 6.4 W
Total SRAM: ~30 W
```

### Step 3: HBM3 interface power

2 HBM3 stacks at 800 GB/s each = 1.6 TB/s total.
Each stack: ~10-12W
```
HBM interface power = 2 * 11 = 22 W
```

Plus the on-chip PHY logic:
```
HBM PHY: ~8 W
Total HBM: ~30 W
```

### Step 4: NoC and control

For a chip with ~100 compute tiles connected by a mesh NoC:
```
NoC power: ~20 W (routers, links, arbitration)
Control logic (schedulers, DMA engines, command processors): ~15 W
Total: ~35 W
```

### Step 5: I/O

PCIe Gen5 x16: ~5 W
Other I/O (chip-to-chip if applicable): ~5 W
```
Total I/O: ~10 W
```

### Step 6: Clock distribution

For a ~300 mm^2 die at 1-1.5 GHz:
```
Clock power: ~25 W (clock tree buffers, mesh clock distribution)
```

### Step 7: Leakage (static power)

At 5nm, leakage is approximately 15-20% of total power for a ~300 mm^2 die:
```
Leakage ≈ 0.18 * TDP = 0.18 * 300 = 54 W
```

### Step 8: Power budget summary

| Component | Power (W) | Fraction |
|---|---|---|
| MAC array + registers | 56 | 18.7% |
| SRAM (64 MB) | 30 | 10.0% |
| HBM3 interface | 30 | 10.0% |
| NoC + control | 35 | 11.7% |
| I/O | 10 | 3.3% |
| Clock distribution | 25 | 8.3% |
| Leakage | 54 | 18.0% |
| **Subtotal** | **240** | **80.0%** |
| **Margin (20%)** | **60** | **20.0%** |
| **Total** | **300** | **100%** |

### Step 9: Feasibility check

The design fits within 300W with 20% margin. Key metrics:

```
TOPS/W = 400 / 300 = 1.33 TOPS/W (competitive for a data center inference chip)
```

Area estimate:
- MAC array: 200T MACs at ~0.08 mm^2 per TOPS = 16 mm^2
  More precisely: 200T/1GHz = 200K MAC units * 80 um^2 = 16 mm^2
- SRAM (64 MB): 64 * 8 / 30 Mbit/mm^2 (5nm) ≈ 17 mm^2
- HBM PHY + controllers: ~15 mm^2
- NoC, control, I/O: ~20 mm^2
- Total active area: ~68 mm^2
- With overhead (routing, power grid, etc.) 2x: ~136 mm^2

A ~150 mm^2 die at 5nm is quite feasible (well within reticle limits). The design is power-limited rather than area-limited, leaving room to add more SRAM or compute if the power budget increases.

### Conclusion

The 400 TOPS / 300W design is feasible at 5nm. The power breakdown shows that the MAC array consumes only 19% of total power -- the majority goes to data movement (SRAM, HBM, NoC), clock distribution, and leakage. This reinforces the importance of architectural optimizations that reduce data movement (tiling, fusion, quantization) over simply adding more MAC units.
