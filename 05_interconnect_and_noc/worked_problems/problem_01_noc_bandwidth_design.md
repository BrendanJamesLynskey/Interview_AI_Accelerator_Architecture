# Worked Problem: NoC Bandwidth Design

## Problem Statement

You are designing the on-chip interconnect for an AI accelerator with:
- 64 processing elements (PEs) in an 8x8 mesh
- Each PE has a 128x128 INT8 systolic array running at 1 GHz (32,768 INT8 ops/cycle)
- A central global buffer (SRAM) of 64 MB at the mesh edge
- Off-chip HBM at 2 TB/s

The target workload is tiled GEMM with K-dimension tile size of 256.

Determine:
1. The per-PE data consumption rate
2. The required link bandwidth in the mesh
3. Whether the global buffer and HBM can sustain the traffic
4. The bisection bandwidth requirement

---

## Worked Solution

### Step 1: Per-PE data consumption rate

Each PE's 128x128 systolic array processes a tile with Tm=128, Tn=128, Tk=256.

Per tile computation:
- Duration: 128 + 128 + 256 - 2 = 510 cycles at 1 GHz = 510 ns
- A tile data: 128 * 256 * 1 byte = 32 KB
- B tile data: 256 * 128 * 1 byte = 32 KB
- Total input data per tile: 64 KB (65,536 bytes)

Data consumption rate per PE:
```
Rate = 65,536 B / 510 ns = 128.5 GB/s
```

### Step 2: Required mesh link bandwidth

With an 8x8 mesh, data flows from the global buffer (at the edge) to PEs throughout the mesh. The worst case is a PE on the far side of the mesh, 7 hops from the buffer edge.

However, data distribution can be optimized: if the global buffer has access ports on all 4 edges of the mesh, the maximum distance to any PE is 4 hops.

The links nearest to the buffer edge carry traffic for all PEs in their column/row:
- An edge link serving 8 PEs in a column: 8 * 128.5 GB/s = 1028 GB/s

This is extremely high. A single mesh link of 128 bits at 1 GHz provides only 16 GB/s.

To achieve 1028 GB/s per edge link: 1028 / 16 = 64 parallel 128-bit links (impractical for a single mesh link).

### Step 3: Redesign with distributed memory

The analysis shows that a central global buffer cannot feed 64 PEs through a mesh -- the edge links become bottlenecks. The solution is distributed memory:

**Option A: Per-PE local SRAM**
- 64 MB / 64 PEs = 1 MB per PE
- Each PE loads tiles from its local SRAM
- Local SRAM bandwidth: essentially unlimited (on-PE access)
- HBM -> local SRAM loading is done via DMA through the mesh at lower urgency

**Option B: Distributed buffer banks**
- Place SRAM banks throughout the mesh (e.g., one bank per 4 PEs)
- 16 banks * 4 MB = 64 MB total
- Each PE accesses its nearest bank (1-2 hops)
- Bank bandwidth requirement: 4 * 128.5 = 514 GB/s per bank = 32 parallel 128-bit links at 1 GHz

Option A (per-PE SRAM) is the standard approach in modern spatial architectures.

### Step 4: HBM to PE loading bandwidth

The 64 PEs collectively consume data at 64 * 128.5 = 8224 GB/s during computation. However, with data reuse (each tile is loaded once and reused across multiple computation passes), the actual HBM bandwidth requirement is much lower.

For a large GEMM (M=N=K=4096) tiled with Tm=Tn=128, Tk=256:
- Total operations: 2 * 4096^3 = 1.374 * 10^11 = 137.4 GOPS
- Total data from HBM (each matrix loaded once, which needs A and B — 33.6 MB — to stay resident in the 64 MB buffer): A (4096^2 bytes) + B (4096^2 bytes) + C (4096^2 * 4 bytes) = 100.7 MB
- Peak compute: each PE does 128 * 128 = 16,384 MACs = 32,768 ops per cycle = 32.77 TOPS at 1 GHz; 64 PEs = 2,097 TOPS
- Compute time: 1.374 * 10^11 / (2.097 * 10^15) = 65.5 us (at peak throughput)

HBM bandwidth during computation: 100.7 MB / 65.5 us = 1.54 TB/s — about 77% of the 2 TB/s available, even with perfect reuse. HBM is therefore close to becoming the bottleneck: any tile reloading (for example, if A and B could not stay resident) would make this GEMM memory-bound.

### Step 5: Bisection bandwidth

The 8x8 mesh bisection cuts through 8 links (cutting the mesh into two 4x8 halves). If each link is 128 bits at 1 GHz = 16 GB/s:
```
Bisection bandwidth = 8 * 16 = 128 GB/s
```

For the distributed SRAM approach, bisection bandwidth is mainly needed for loading new data from HBM (which enters at the edges) to PEs. From Step 4, HBM data arrives at about 1.5 TB/s, and roughly half of it must cross the bisection to reach the far half of the mesh — about 0.77 TB/s, six times the 128 GB/s available. The design therefore needs wider links, HBM/buffer ports on several edges, or more on-chip reuse so that less data crosses the mesh.

### Key Takeaway

The critical lesson is that a centralized memory with a mesh interconnect cannot sustain the data rates of a large PE array. Distributed memory (per-PE or per-cluster SRAM) is essential, and the mesh interconnect then carries mainly HBM refill traffic, results, and control signalling. As Step 5 shows, that refill traffic is still large enough that the mesh bisection must be sized for it.
