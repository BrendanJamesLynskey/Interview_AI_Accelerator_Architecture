# HBM and Off-Chip Memory

This section covers High Bandwidth Memory (HBM) technology, die stacking, channel architecture, and the evolution from HBM2e through HBM3e for AI accelerators.

---

### Q1. What is HBM and how does die stacking enable high bandwidth?

**Answer:**

High Bandwidth Memory (HBM) is a 3D-stacked DRAM technology that achieves high bandwidth by vertically stacking multiple DRAM dies on top of a base logic die and connecting them through thousands of Through-Silicon Vias (TSVs). HBM is co-packaged with the processor die on a silicon interposer using 2.5D integration.

Traditional DRAM (DDR) communicates with the processor through a narrow interface (64 bits per channel) over long PCB traces. The pin count and trace length limit bandwidth to approximately 50-100 GB/s per channel, and the long traces consume significant power for I/O.

HBM achieves dramatically higher bandwidth through:
1. **Wide interface**: Each HBM stack has 1024 data pins (compared to 64 for a DDR channel), organized as 8 or 16 independent channels of 64 or 128 bits each. This 16x wider interface directly translates to 16x more data per clock cycle.
2. **Short interconnections**: TSVs within the stack are only tens of micrometers long, and the interposer traces from HBM to the processor are millimeters rather than centimeters. This enables higher signaling rates at lower power per bit.
3. **Low power**: The short interconnect distances reduce I/O power by approximately 3-5x compared to DDR DIMM connections.

The manufacturing process involves: (a) fabricating individual DRAM dies, (b) thinning them to approximately 50 micrometers, (c) stacking them using micro-bumps aligned with TSVs, (d) mounting the stack on a base die (which contains the control logic and PHY), and (e) placing the completed stack on a silicon interposer alongside the processor die.

Each additional DRAM die in the stack increases capacity proportionally. HBM2 stacks 4 or 8 dies; HBM3 stacks 8 or 12 dies; HBM3e stacks up to 12 dies for capacities up to 36 GB per stack.

---

### Q2. How has HBM evolved from HBM2e to HBM3 to HBM3e?

**Answer:**

| Specification | HBM2e | HBM3 | HBM3e |
|---|---|---|---|
| Data rate per pin | 3.6 Gbps | 6.4 Gbps | 9.6 Gbps |
| Channels per stack | 8 (128-bit each) | 16 (64-bit each) | 16 (64-bit each) |
| Total bus width | 1024 bits | 1024 bits | 1024 bits |
| Bandwidth per stack | 460 GB/s | 819 GB/s | 1.2 TB/s |
| Capacity per stack (typical) | 16 GB (8-Hi) | 16-24 GB (8-12 Hi) | 24-36 GB (8-12 Hi) |
| Die stacking | 4-8 Hi | 8-12 Hi | 8-12 Hi |
| Used in | A100, MI250 | H100, MI300X | B200, MI350 |

Key evolutionary trends:

**HBM2e to HBM3**: The major change was doubling the number of channels from 8 to 16 while maintaining the 1024-bit total bus width. Each channel narrowed from 128 bits to 64 bits. This enables finer-grained access (smaller minimum access granularity) and independent operation of more channels, improving utilization for workloads with diverse access patterns. The per-pin data rate also increased from 3.6 to 6.4 Gbps.

**HBM3 to HBM3e**: The primary improvement is a 50% increase in per-pin data rate (6.4 to 9.6 Gbps), directly translating to 50% more bandwidth per stack. Capacity also increased through higher die stacking (12-Hi) and denser per-die capacity.

For the B200 with 8 HBM3e stacks: 8 * 1.2 TB/s = 8+ TB/s total bandwidth, 8 * 24 GB = 192 GB total capacity. This represents a 4x bandwidth improvement over the A100 (2.0 TB/s) in just two generations.

---

### Q3. What is a silicon interposer and why is it needed for HBM?

**Answer:**

A silicon interposer is a thin silicon substrate that acts as a high-density wiring layer between the processor die and the HBM stacks. It is placed between the dies and the organic package substrate, providing a finer-pitch routing medium than the package substrate alone can support.

The interposer is needed because:

1. **Pitch mismatch**: The micro-bump pitch on HBM stacks (approximately 40-55 um) is much finer than what organic package substrates can reliably route (approximately 100-150 um line/space). The silicon interposer, fabricated with standard semiconductor lithography, can route at pitches as fine as 1-10 um, bridging this gap.

2. **Signal integrity**: The short, well-controlled traces on the silicon interposer provide better signal integrity than longer organic package traces, enabling higher data rates with lower power.

3. **Multiple HBM stacks**: A modern GPU may have 5-8 HBM stacks, each with 1024 data signals. Routing 5000-8000+ signals from the processor die to the HBM stacks through an organic substrate would be prohibitively complex. The interposer provides the wiring density needed.

The interposer itself is typically fabricated at an older process node (65nm or 45nm) since it contains only passive wiring (no transistors, or minimal transistors for TSVs). It uses TSVs to connect the top-side micro-bumps (facing the processor and HBM) to the bottom-side C4 bumps (facing the package substrate).

The main disadvantage of silicon interposers is cost. A large interposer (for a chip with many HBM stacks) can be as expensive as a small processor die. This has motivated alternatives such as EMIB (Intel's Embedded Multi-Die Interconnect Bridge), which embeds small silicon bridges in the package substrate only where needed, and organic interposers with higher-density routing.

TSMC's CoWoS (Chip-on-Wafer-on-Substrate) is the dominant interposer packaging technology for AI accelerators. CoWoS-S uses a full silicon interposer; CoWoS-R uses a redistribution layer; CoWoS-L uses a local silicon bridge combined with an organic interposer.

---

### Q4. How does HBM channel architecture affect access patterns and performance?

**Answer:**

HBM3 organizes its 1024-bit interface into 16 independent channels, each 64 bits wide. Understanding how data is distributed across channels is important for maximizing bandwidth utilization.

**Address interleaving**: The memory controller typically interleaves addresses across channels so that consecutive cache lines are distributed across different channels. With 16 channels, every 16th cache line goes to the same channel. This ensures that sequential access patterns (common in AI) utilize all channels equally.

**Channel independence**: Each channel operates independently with its own command bus, read/write queues, and timing constraints. This means that one channel can be performing a read while another performs a write, and different channels can serve different bank rows simultaneously.

**Minimum access granularity**: Each HBM3 channel transfers 64 bits per beat. At 6.4 Gbps, this is 64 bits * 6.4 * 10^9 / 8 = 51.2 GB/s per channel. The minimum useful access is one burst (typically 32 bytes per channel). Accessing less than 32 bytes wastes bandwidth.

**Bank structure**: Each channel contains multiple banks (typically 16 banks per HBM3 channel). Banks within a channel share the channel's data bus but can have different rows open simultaneously. Row buffer hits (accessing data in an already-open row) have lower latency than row buffer misses (which require a precharge and activate cycle).

**Performance implications for AI**:
- Large sequential reads (loading weight matrices) achieve near-peak bandwidth because they utilize all channels and exploit row buffer locality.
- Small random reads (embedding table lookups) may underutilize bandwidth because they cause row buffer misses and may not distribute evenly across channels.
- Strided access patterns (reading a column of a row-major matrix) may create hot channels if the stride aligns with the interleaving granularity, reducing effective bandwidth.

The memory controller must balance row buffer management (keeping rows open for temporal locality) with channel utilization (distributing accesses across channels). For AI training workloads with large, sequential data access patterns, this balance is relatively easy to achieve.

---

### Q5. What is the power consumption of HBM and how does it compare to computation power?

**Answer:**

HBM power consumption consists of several components:

**DRAM core power**: The power consumed by the DRAM arrays (row activation, sense amplifiers, bitline precharge). This scales with the number of bank activations per second and is largely independent of data rate.

**I/O power**: The power consumed by the I/O drivers and receivers transmitting data across the micro-bumps and interposer traces. This scales with data rate and bus width.

**Refresh power**: DRAM cells lose charge over time and must be periodically refreshed (typically every 32-64 ms for the entire array). This is a baseline power that is always present.

**Total HBM power**: For a modern HBM3 stack operating at full bandwidth, total power is approximately 10-15W per stack. With 5 stacks (as in H100), HBM power is approximately 50-75W, which is 7-11% of the chip's 700W TDP.

Comparison with compute power:
- The H100's tensor cores consume roughly 300-400W at full utilization.
- HBM consumes 50-75W at full bandwidth.
- The remaining 225-350W goes to I/O, NoC, control logic, and leakage.

The HBM power fraction is relatively modest, but it is important to note that HBM bandwidth cannot be improved simply by adding more power. The bandwidth is constrained by the pin count, data rate per pin, and channel count, all of which have physical and technology limits.

For future architectures, as compute density continues to increase faster than memory bandwidth, the energy cost of data movement (from HBM to the compute units) becomes an increasingly large fraction of total energy. This reinforces the importance of on-chip SRAM and data reuse: accessing data from local SRAM costs 50-100x less energy per bit than accessing HBM.

---

### Q6. What are the alternatives to HBM for AI accelerator memory?

**Answer:**

**GDDR6/GDDR6X**: The standard memory for consumer GPUs. GDDR6X offers up to 1 TB/s bandwidth (e.g., RTX 4090 with 384-bit bus at 21 Gbps). Lower cost per bit than HBM but lower bandwidth density (bandwidth per package footprint) and higher power per bit. Used in some inference accelerators where cost matters more than peak bandwidth.

**LPDDR5/LPDDR5X**: Low-power DRAM used in mobile and edge devices. Apple's M-series chips use LPDDR5 unified memory at up to 400 GB/s. Lower bandwidth than HBM but much lower power consumption and simpler packaging (soldered directly to the package substrate, no interposer needed).

**On-chip SRAM**: Cerebras and Graphcore eliminate off-chip memory entirely, using only distributed on-chip SRAM. This provides the highest bandwidth (tens of PB/s aggregate) at the lowest access energy but at the highest cost per bit and lowest capacity (tens of GB maximum).

**CXL-attached memory**: Compute Express Link (CXL) enables coherent access to memory pools beyond the local HBM. This can extend effective memory capacity to terabytes at the cost of higher latency and lower bandwidth than HBM. Useful for inference of very large models that do not fit in HBM.

**Processing-in-memory (PIM)**: Approaches like Samsung's HBM-PIM add simple compute logic within the HBM stack itself, enabling some operations (like embedding lookups or activation functions) to be performed within the memory, eliminating the bandwidth bottleneck for those operations.

**3D-stacked SRAM**: Some designs stack SRAM layers on top of the logic die using hybrid bonding, providing more on-chip memory capacity without consuming logic die area. This is being explored for future accelerators.

Each alternative occupies a different point in the capacity-bandwidth-energy-cost design space. HBM dominates for high-end training and inference accelerators because it provides the best bandwidth density for the power budget. But the alternatives are important for edge inference, very large model serving, and future architectures.

---

### Q7. How does HBM bandwidth interleaving work across stacks?

**Answer:**

Bandwidth interleaving distributes data across multiple HBM stacks (and channels within stacks) so that consecutive memory accesses utilize different stacks, maximizing aggregate bandwidth.

A typical interleaving scheme on an H100 with 5 HBM3 stacks (80 total channels):

```
Physical address -> Stack and channel mapping:
  Stack = (address / cache_line_size) % num_stacks
  Channel within stack = ((address / cache_line_size) / num_stacks) % channels_per_stack
```

With 64-byte cache lines, 5 stacks, and 16 channels per stack:
- Byte addresses 0-63 go to stack 0, channel 0
- Byte addresses 64-127 go to stack 1, channel 0
- Byte addresses 128-191 go to stack 2, channel 0
- ...
- Byte addresses 256-319 go to stack 0, channel 1
- ...

This ensures that a sequential read of a large tensor (weight matrix, activation tensor) spreads evenly across all stacks and channels, achieving the full aggregate bandwidth (5 * 670 GB/s = 3.35 TB/s for H100).

The interleaving granularity is chosen to balance two concerns:
1. **Too fine** (byte-level interleaving): Each HBM access pulls a minimum burst of data, and interleaving at finer granularity wastes the rest of the burst.
2. **Too coarse** (megabyte-level interleaving): Large contiguous allocations would reside on a single stack, creating bandwidth hotspots.

Cache-line-level interleaving (64-128 bytes) is the typical choice, as it matches the minimum useful HBM burst size and ensures good distribution for most access patterns.

The memory controller must also handle address mapping for non-power-of-two stack counts (5 stacks on H100). This requires modular arithmetic rather than simple bit extraction, adding complexity but ensuring even distribution.

---

### Q8. What is the impact of HBM latency versus bandwidth for AI workloads?

**Answer:**

HBM access latency is approximately 100-150 ns (from command to first data returned), which corresponds to roughly 400-600 GPU clock cycles at 2 GHz. After the initial latency, data streams at the full bandwidth rate.

For AI workloads, bandwidth is almost always more important than latency for the following reasons:

**Large access sizes**: AI workloads access large, contiguous data structures (weight matrices, activation tensors, gradient buffers). A single weight matrix might be 32 MB. The time to transfer 32 MB at 3.35 TB/s is 9.6 microseconds, of which the initial 0.15 microseconds of latency is negligible.

**Parallelism hides latency**: GPUs run thousands of concurrent threads (warps). While one warp stalls waiting for its HBM read to return, other warps continue executing. With sufficient parallelism (high occupancy), the latency is completely hidden. This is the fundamental GPU latency-hiding mechanism.

**Predictable access patterns**: AI access patterns are predictable, allowing prefetching to initiate HBM reads well before the data is needed. Double buffering further hides latency by overlapping data loading with computation.

However, latency matters in some specific scenarios:
- **Small-batch inference**: With batch size 1, there may not be enough parallel work to hide memory latency. Each layer's computation completes quickly, and the latency to start loading the next layer's weights cannot be fully hidden.
- **Irregular access patterns**: Embedding table lookups with random indices cannot benefit from prefetching or pipelining, making each lookup pay the full latency.
- **Synchronization barriers**: After a synchronization point (e.g., end of a layer), all work that depends on the synchronized result must wait for the memory access, and there may not be independent work to hide the latency.

For accelerator design, the bandwidth-over-latency priority means that HBM technology development focuses on increasing data rate per pin and widening the bus, rather than reducing access latency. Latency improvements are a secondary benefit that comes naturally from shorter interconnect distances in advanced packaging.

---

### Q9. How does ECC in HBM affect capacity and performance?

**Answer:**

Error Correcting Code (ECC) adds redundant bits to each data word, allowing the detection and correction of single-bit errors and the detection of multi-bit errors. In data center HBM, ECC is essential for ensuring data integrity during long training runs where a single bit flip could corrupt a model.

**Capacity impact**: ECC typically adds approximately 12.5% overhead (128 data bits + 16 ECC bits = 144 total bits per 128-bit word). For HBM with 16 GB of raw DRAM capacity, approximately 14.2 GB is usable for data and 1.8 GB is consumed by ECC bits. The "80 GB" H100 HBM capacity refers to the usable data capacity after ECC overhead.

**Performance impact**: ECC computation (encoding on writes, syndrome generation and correction on reads) adds a small amount of latency (typically 1-2 ns) and power consumption. The bandwidth impact is minimal because the ECC bits share the same data bus as the data -- they are stored inline within the HBM array and read/written as part of the same burst. However, the ECC bits do consume storage capacity that could otherwise hold data.

**Scrubbing**: To prevent accumulated errors from overwhelming the ECC capability (a single-bit error that is not corrected can accumulate into a multi-bit error over time), the memory controller performs periodic scrubbing: reading all memory locations, checking ECC, correcting any single-bit errors, and rewriting the corrected data. Scrubbing consumes a small amount of bandwidth (typically less than 1%) and is scheduled during low-utilization periods.

**Consumer vs data center**: Consumer GPUs (GeForce/RTX series) typically do not enable ECC on GDDR memory, using the full capacity for data. Data center GPUs (A100, H100) always enable ECC on HBM. This partly explains why consumer GPU memory specifications sometimes show higher capacity or bandwidth than data center GPUs using the same memory technology.

---

### Q10. What is the future trajectory of memory technology for AI accelerators?

**Answer:**

Several technology trends will shape AI accelerator memory in the coming years:

**HBM4 (expected 2025-2026)**: JEDEC standardization is ongoing. Expected improvements include 12-16 Hi stacking (increasing per-stack capacity to 48-64 GB), data rates up to 12-16 Gbps per pin, and potentially a wider interface (2048 bits per stack). Bandwidth per stack could reach 2+ TB/s. Logic integration within the HBM base die (allowing in-memory compute) is being explored.

**Hybrid bonding**: This packaging technology enables much finer-pitch die-to-die connections (1-10 um pitch vs 40 um for micro-bumps), dramatically increasing the interconnect density between stacked dies. This could enable SRAM stacking on logic, higher HBM bandwidth, and new memory architectures.

**CXL memory pooling**: CXL 3.0 enables shared memory pools across multiple accelerators, effectively creating a distributed memory tier with higher capacity but lower bandwidth than local HBM. This could enable serving models with trillions of parameters across CXL-connected memory nodes.

**Processing-in-memory (PIM)**: Adding compute capabilities within or near the memory arrays. Samsung, SK Hynix, and others are developing PIM solutions that can perform operations like multiply-accumulate within the HBM stack, reducing the need to move data to the processor. This is particularly promising for memory-bound operations.

**Photonic interconnects**: Optical interconnects could provide much higher bandwidth over longer distances at lower power than electrical signaling. This could enable disaggregated memory architectures where a large memory pool is connected to the processor via photonic links at near-HBM bandwidth.

**Non-volatile memory**: Technologies like STT-MRAM and resistive RAM could provide persistent, non-volatile on-chip or near-chip memory for storing model weights that do not change during inference. This would eliminate the need to load weights from HBM at startup.

The common theme is that the memory wall will persist, and the industry is pursuing multiple approaches to address it through packaging innovation, new memory technologies, and architectural redesign of the processor-memory interface.
