# On-Chip Interconnect

This section covers on-chip interconnect architectures for AI accelerators, including buses, crossbars, and flow control mechanisms.

---

### Q1. Why is the on-chip interconnect critical for AI accelerator performance?

**Answer:**

The on-chip interconnect is the communication fabric that connects compute units to memory (SRAM, HBM controllers) and to each other. In an AI accelerator with hundreds of compute units (SMs, PEs, tensor cores) and multiple memory banks, the interconnect determines how quickly data can be moved between producers and consumers. If the interconnect cannot supply data at the rate the compute units consume it, the compute units stall and utilization drops.

For a systolic array, the interconnect between the global buffer and the PE array must deliver a new operand to each PE every cycle. For a 256x256 array at 1 GHz, this requires 256 * 2 (two operands per PE) * 1 byte = 512 GB/s of bandwidth from the buffer to the array edges alone. The internal PE-to-PE interconnect (nearest-neighbor wires in the systolic array) provides additional bandwidth through data forwarding.

For a GPU, the interconnect includes: the crossbar connecting SMs to L2 cache partitions, the L2-to-HBM controller links, the SM-internal shared memory bus, and the inter-SM network for global memory access. The H100's L2 cache provides ~12 TB/s of bandwidth to the SMs through a wide crossbar network.

Interconnect design involves tradeoffs between bandwidth, latency, area, power, and scalability. A full crossbar provides maximum bandwidth but scales quadratically in area (O(N^2) for N ports). A mesh network scales linearly but has higher latency for distant nodes. The optimal choice depends on the communication patterns of the target workload and the number of connected components.

---

### Q2. What are the main on-chip interconnect topologies?

**Answer:**

**Bus**: A shared communication channel connecting all nodes. Simple and low-area but bandwidth is shared: as more nodes are added, the per-node bandwidth decreases. Suitable for small systems (4-8 nodes) with moderate bandwidth requirements. Not used in modern AI accelerators for the main interconnect due to poor scalability.

**Crossbar**: A non-blocking switch that provides a dedicated path between any input-output pair. Every pair can communicate simultaneously at full bandwidth. Area scales as O(N^2) and power scales similarly, making it impractical for large port counts (beyond ~32-64). Used in GPU L2-to-SM interconnects where the number of L2 partitions is modest.

**Mesh/Torus**: A 2D grid where each node connects to its 4 nearest neighbors (mesh) or wraps around edges (torus). Area scales linearly with node count. Bandwidth between non-adjacent nodes requires multi-hop routing, adding latency (proportional to Manhattan distance). Widely used in large-scale accelerators: TPU ICI uses a torus, Cerebras WSE uses a 2D mesh.

**Ring**: Each node connects to its two neighbors in a ring. Simple and low-area but bandwidth between distant nodes requires traversing half the ring. Used within NVIDIA NVLink systems (ring-based all-reduce).

**Tree/Fat Tree**: Hierarchical tree with increasing bandwidth toward the root. Provides logarithmic latency but potential bandwidth bottleneck at the root. Fat trees (where bandwidth increases toward the root) address this but add area.

**Hierarchical/Hybrid**: Modern accelerators combine topologies at different scales. For example, a crossbar within an SM cluster connecting to a mesh between clusters. This balances the scalability of mesh with the low-latency of crossbar at the local level.

---

### Q3. What flow control mechanisms are used in on-chip networks?

**Answer:**

Flow control manages how data traverses the network when resources (buffers, links) are contested. The primary mechanisms are:

**Credit-based flow control**: Each downstream node allocates buffer space and sends credits (tokens indicating available buffer slots) upstream. The upstream node can send data only when it holds a credit. When the downstream node frees a buffer slot, it returns the credit. This prevents buffer overflow at the cost of storing credits and credit return latency.

**Virtual channels (VCs)**: Multiple logical channels share the same physical link. Each VC has its own buffer queue and flow control. VCs are essential for avoiding deadlock (by ensuring that messages of different priorities or routing classes use separate VCs) and for improving throughput (a blocked message in one VC does not block messages in other VCs).

**Wormhole routing**: Messages are divided into flits (flow control units). The header flit establishes the route, and body flits follow through the same path. Only the header flit needs routing logic; body flits follow automatically. Wormhole routing reduces buffer requirements (only a few flits per node need buffering) but can block other messages if a message stalls mid-route (head-of-line blocking).

**Store-and-forward**: Each node receives and buffers the entire packet before forwarding. This avoids head-of-line blocking but adds latency equal to the packet size and requires large buffers.

For AI accelerators, credit-based flow control with wormhole routing and 2-4 virtual channels is the most common design. The regular, predictable communication patterns of AI workloads (tiled data transfers, deterministic schedules) simplify flow control compared to general-purpose NoC designs.

---

### Q4. How is the interconnect between SMs and L2 cache organized in a modern GPU?

**Answer:**

In NVIDIA's Hopper architecture (H100), the 132 SMs connect to the L2 cache through a partition-based crossbar:

The L2 cache is divided into partitions (approximately 32 partitions, each ~1.5 MB, totaling ~50 MB). Each partition is associated with a memory controller and a slice of the address space. The crossbar connects all SMs to all L2 partitions, allowing any SM to access any L2 partition.

Address interleaving distributes consecutive cache lines across L2 partitions, so a sequential memory access from an SM hits different partitions on successive accesses. This spreads the load and maximizes aggregate bandwidth.

The crossbar is not a full NxN crossbar (132 SMs x 32 L2 partitions would require 4,224 crosspoints). Instead, it is typically implemented as a hierarchical or segmented crossbar where SMs are grouped into clusters that share a local switch, and these local switches connect to the L2 partitions through a second level of switching.

Each L2 partition connects to one or more HBM channels. On an L2 miss, the request flows from the L2 partition to the HBM controller, which issues the DRAM command and returns the data through the same path.

The aggregate bandwidth from L2 to SMs is approximately 12 TB/s, roughly 4x the HBM bandwidth (3.35 TB/s). This amplification factor means that data reused from L2 (hit by multiple SMs or multiple accesses from the same SM) is served 4x faster than a fresh HBM load.

---

### Q5. What is bandwidth provisioning and how do architects determine interconnect bandwidth?

**Answer:**

Bandwidth provisioning is the process of determining how much communication bandwidth each link, switch, and network segment needs to avoid becoming a performance bottleneck. The goal is to provision enough bandwidth to keep the compute units fed without over-provisioning (which wastes area and power).

The analysis starts from the compute units' data consumption rate:

```
Required bandwidth = compute_throughput / arithmetic_intensity_of_target_workload
```

For a 256x256 systolic array at 1 GHz processing a weight-stationary GEMM:
- The array performs 256^2 = 65,536 MACs/cycle
- With weight-stationary dataflow, 256 new activations enter from the left every cycle and 256 partial results exit at the bottom
- Required input bandwidth: 256 elements/cycle * 2 bytes = 512 bytes/cycle = 512 GB/s
- Required output bandwidth: 256 elements/cycle * 4 bytes (FP32 accum) = 1024 bytes/cycle = 1024 GB/s

The interconnect from the global buffer to the array edges must support this bandwidth. If the global buffer has 32 banks, each bank must provide 512/32 = 16 GB/s = 16 bytes/cycle, which is a single 128-bit read per cycle at 1 GHz -- feasible.

Additional considerations:
- **Contention**: If multiple compute units share the same interconnect (as in a GPU with many SMs sharing a crossbar), the provisioned bandwidth must account for worst-case contention patterns.
- **Bisection bandwidth**: The minimum bandwidth across any cut that divides the network in half. This is the theoretical maximum all-to-all bandwidth and should be at least half the aggregate compute-to-memory bandwidth.
- **Overhead**: Protocol headers, flow control credits, and error correction reduce the usable fraction of raw bandwidth (typically 80-90% efficiency).

---

### Q6. How does the interconnect design differ between training and inference accelerators?

**Answer:**

Training and inference workloads impose different demands on the on-chip interconnect:

**Training accelerators** require high bandwidth in both directions (forward and backward pass data), support for gradient accumulation (reduce operations across compute units), and efficient broadcast of shared data (weights to all compute units, activations for data parallelism). The interconnect must support both multicast (one-to-many for broadcasting) and reduction (many-to-one for gradient accumulation) patterns efficiently. The data volumes are large (full batch activations and gradients) and the communication is relatively predictable.

**Inference accelerators** typically process smaller batch sizes and have asymmetric bandwidth requirements. Weight loading (memory-to-compute) dominates bandwidth in most layers. The interconnect must efficiently support streaming weights from memory to compute units with minimal latency. For low-latency inference, the interconnect latency (number of hops, queuing delays) becomes more critical because there is less computation to hide behind.

Specific design differences:
- Training accelerators may include dedicated reduction trees or reduction networks for gradient accumulation (partial sums from different compute units are combined on-chip before being written to memory).
- Inference accelerators may include broadcast trees for efficient weight distribution to multiple compute units processing different batch elements.
- Multi-chip interconnect (NVLink, ICI) is critical for training (gradient synchronization across chips) but less important for single-chip inference.

---

### Q7. What is the role of the NoC in a chiplet-based accelerator?

**Answer:**

As accelerator dies grow larger and approach the reticle limit (~800 mm^2), chiplet-based designs partition the functionality across multiple smaller dies connected through an interposer or advanced packaging. The NoC (Network-on-Chip) in a chiplet design has two levels:

**Intra-chiplet NoC**: The on-chip network within each chiplet, connecting local compute units to local SRAM and local I/O interfaces. This is similar to a monolithic design's NoC but smaller and potentially simpler.

**Inter-chiplet interconnect**: The communication fabric that connects chiplets through the package. This can be implemented through: silicon bridge interconnects (like Intel EMIB), through-silicon interposer wiring (like TSMC CoWoS), or organic substrate traces.

The inter-chiplet interconnect has several key challenges compared to on-chip wires:
- **Higher latency**: Inter-chiplet links traverse longer distances (mm to cm vs um on-chip) through higher-resistance paths (interposer or package traces vs on-chip metal).
- **Lower bandwidth density**: Interposer bump pitch (~40 um) is much coarser than on-chip wire pitch (~1 um), limiting the number of parallel connections.
- **Higher power**: Driving signals across chiplet boundaries requires I/O drivers that consume more power per bit than on-chip data movement.

For AI accelerators, the chiplet approach is motivated by yield improvement (smaller dies have higher yield), flexibility (mixing different process nodes for compute, memory, and I/O chiplets), and cost reduction. AMD's MI300X uses chiplets extensively, and NVIDIA's B200 uses a dual-die design. The NoC architecture must be designed to hide the inter-chiplet latency and bandwidth limitations, typically by ensuring that most communication stays local within a chiplet and only inter-chiplet communication crosses the boundary for global operations like reduction and synchronization.

---

### Q8. How do multicast and reduction operations map onto the on-chip interconnect?

**Answer:**

**Multicast** (one-to-many): A single data item is sent from one source to multiple destinations. In AI accelerators, multicast is used for broadcasting weight tiles to multiple compute units, broadcasting input activations to multiple output channel computations, and distributing control signals.

In a mesh network, multicast is implemented by replicating packets at intermediate nodes: the header specifies the set of destinations, and each node forwards copies toward the appropriate destinations. In a tree network, multicast naturally follows the tree structure (data flows from root to leaves). In a crossbar, multicast requires the crossbar to simultaneously connect one input to multiple outputs, which may require additional buffering if outputs operate at different rates.

**Reduction** (many-to-one): Multiple data items from different sources are combined (summed, max-ed, etc.) into a single result at one destination. In AI accelerators, reduction is used for accumulating partial sums from different compute units (in tensor parallelism), gradient aggregation, and pooling operations.

Hardware-accelerated reduction is valuable because software-based reduction (sending all values to one node for serial addition) creates a bandwidth bottleneck at the destination. A reduction tree distributes the addition across intermediate nodes: each node adds two inputs and forwards the result upward. An N-input reduction tree completes in O(log N) steps instead of O(N) for serial reduction.

Some accelerators include dedicated reduction logic at network switches or crossbar ports. For example, NVIDIA's NVSwitch includes on-chip reduction support for all-reduce operations, allowing gradient values from different GPUs to be summed as they pass through the switch without requiring a separate reduction phase.

---

### Q9. What is the power cost of on-chip data movement compared to computation?

**Answer:**

On-chip data movement is a significant and often dominant component of total power consumption:

| Operation | Energy (7nm, approximate) |
|---|---|
| INT8 multiply | 0.1 pJ |
| FP16 multiply | 0.4 pJ |
| FP32 multiply | 1.5 pJ |
| Read 16 bits from register file | 0.5 pJ |
| Read 16 bits from SRAM (local) | 5 pJ |
| Read 16 bits from SRAM (global, 1mm wire) | 10-20 pJ |
| Read 16 bits from HBM | 200 pJ |

Key observations:
- Reading a 16-bit value from a 1mm on-chip wire costs more energy than an FP16 multiply. For a large die (20mm across), moving data from one side to the other costs 10-20x the energy of computing with it.
- SRAM access energy is dominated by the wire length to the SRAM bank, not the SRAM cell itself. Distributed SRAM (small banks close to each compute unit) is more efficient than centralized SRAM (large banks far from compute units).
- HBM access costs roughly 200x more energy per bit than register access.

These energy ratios explain several architectural trends:
- Spatial architectures (systolic arrays, dataflow) minimize data movement by placing compute units adjacent to the data they consume.
- Distributed SRAM (like Cerebras WSE's per-core scratchpads) reduces access energy compared to centralized buffers.
- Operator fusion keeps intermediate results in registers or local SRAM, avoiding costly global memory accesses.
- The proliferation of on-chip SRAM (despite its area cost) is justified by the energy savings of avoiding HBM accesses.

---

### Q10. How do architects validate that an interconnect design meets performance requirements?

**Answer:**

Interconnect validation uses multiple methods at different stages of the design process:

**Analytical modeling**: Early in the design, architects use closed-form models to estimate bandwidth requirements and latency. For example: `required_bandwidth = compute_throughput * bytes_per_op / arithmetic_intensity`. These models identify gross mismatches between compute capability and interconnect bandwidth.

**Cycle-accurate simulation**: Detailed NoC simulators (BookSim, Garnet in gem5, custom RTL simulators) model every flit, every buffer, and every arbitration decision. Traffic patterns from representative AI workloads are injected, and the simulator reports throughput, latency, and contention hotspots. This is the primary validation method for interconnect design.

**Trace-driven simulation**: Real execution traces from AI workloads (captured from profiling runs on existing hardware or from functional simulators) are replayed through the NoC simulator. This provides more realistic traffic patterns than synthetic benchmarks.

**RTL simulation**: Once the NoC design is implemented in RTL (Verilog/SystemVerilog), gate-level simulation verifies correctness and provides accurate timing. However, RTL simulation is too slow for long-running workload traces, so it is used primarily for corner cases and protocol verification.

**FPGA prototyping**: The NoC can be prototyped on an FPGA to run workloads at near-real-time speed (typically 10-100x slower than silicon). This enables testing with complete software stacks and realistic workloads.

Key metrics validated:
- **Sustained bandwidth**: aggregate and per-link, under realistic traffic
- **Latency distribution**: average and tail latency (P99, P99.9)
- **Contention/hotspots**: links or switches that become bottlenecks
- **Deadlock freedom**: formal or exhaustive testing that the routing algorithm and flow control do not produce deadlock
- **Power**: estimated from switching activity captured during simulation
