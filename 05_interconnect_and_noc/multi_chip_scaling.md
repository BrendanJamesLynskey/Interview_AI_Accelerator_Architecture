# Multi-Chip Scaling

This section covers chip-to-chip interconnects, multi-chip architectures, wafer-scale design, and scale-out networking for AI accelerators.

---

### Q1. Why is multi-chip scaling necessary for AI workloads?

**Answer:**

Multi-chip scaling is necessary because the compute and memory requirements of frontier AI models exceed what any single chip can provide. A 175B parameter model in FP16 requires 350 GB just for the weights, far exceeding any single chip's HBM capacity (80-192 GB). Training such a model requires petaFLOPS-months of compute, which can only be achieved by hundreds or thousands of accelerators working in parallel.

Even if a single chip could hold the model, the training throughput of one chip (sustained 500-1000 TFLOPS) would make training prohibitively slow. Data-parallel training divides the batch across chips, and model-parallel training divides the model across chips, both requiring inter-chip communication.

The challenge is that inter-chip bandwidth is 10-1000x lower than on-chip bandwidth. NVLink provides 900 GB/s between GPUs (H100), compared to 3.35 TB/s HBM bandwidth and 12+ TB/s L2 bandwidth on-chip. InfiniBand provides 50-100 GB/s between nodes. This bandwidth hierarchy means that the efficiency of multi-chip scaling depends critically on the interconnect technology and the communication patterns of the parallelism strategy.

---

### Q2. What are the main parallelism strategies for distributed training?

**Answer:**

**Data parallelism**: Each chip holds a complete copy of the model and processes different batch elements. After the backward pass, gradients are synchronized across all chips via all-reduce. Communication volume: 2 * model_size * (N-1)/N for ring all-reduce with N chips. Works well when the model fits on one chip and the batch is large enough.

**Tensor parallelism (TP)**: Each linear layer's weight matrix is partitioned across chips (column-wise or row-wise). Each chip computes its portion and the results are combined via all-reduce or all-gather. Communication per layer: one all-reduce of activation_size. Requires very high inter-chip bandwidth because communication occurs between every layer.

**Pipeline parallelism (PP)**: Different layers are assigned to different chips. Activations are passed forward and gradients backward between pipeline stages. Communication per micro-batch: activation_size between adjacent stages. Lower bandwidth requirement than TP but introduces pipeline bubbles and requires careful scheduling (GPipe, PipeDream).

**Expert parallelism (EP)**: In Mixture-of-Experts models, different experts are placed on different chips. Tokens are routed to the appropriate expert via all-to-all communication. Communication volume depends on the number of tokens routed to each expert.

**ZeRO (Zero Redundancy Optimizer)**: Partitions optimizer states, gradients, and/or parameters across chips (ZeRO Stage 1/2/3). Reduces per-chip memory at the cost of all-gather communication to reconstruct full parameters before each forward/backward pass.

In practice, large-scale training uses a combination of these strategies. A common recipe for GPT-scale training: TP within a node (8 GPUs on NVLink), PP across nodes, DP across pipeline groups, with ZeRO for memory optimization.

---

### Q3. How does NVLink compare to PCIe and InfiniBand for multi-GPU communication?

**Answer:**

| Feature | PCIe Gen5 x16 | NVLink 4.0 (H100) | InfiniBand NDR |
|---|---|---|---|
| Bandwidth (bidirectional) | 128 GB/s | 900 GB/s | 100 GB/s (per port) |
| Latency | ~1-2 us | ~0.5-1 us | ~1-2 us (switch) |
| Reach | Within server | Within server/NVLink domain | Data center |
| Protocol | PCIe (CPU-centric) | NVIDIA proprietary | RDMA/RoCE |
| GPU-to-GPU direct | No (through CPU) | Yes | Yes (GPUDirect RDMA) |
| Scalability | Point-to-point | 256 GPUs (via NVLink Switch) | Thousands of nodes |

NVLink provides 7-18x more bandwidth than PCIe and is essential for tensor parallelism, where activation tensors must be exchanged between GPUs every layer. The latency advantage is moderate but meaningful for small message sizes.

InfiniBand extends connectivity beyond a single server to the entire data center. NDR InfiniBand provides 400 Gb/s (50 GB/s) per port, with multiple ports per GPU (typically 8 ports per H100 in DGX systems, providing 400 GB/s aggregate). The combination of NVLink within a node and InfiniBand between nodes creates a hierarchical bandwidth structure that parallelism strategies must account for.

GPUDirect RDMA enables InfiniBand NICs to read/write GPU memory directly without going through the CPU, reducing latency and CPU overhead for inter-node GPU communication. GPUDirect P2P enables direct GPU-to-GPU transfers over PCIe without CPU involvement (but at PCIe bandwidth).

---

### Q4. What is the Cerebras Wafer-Scale Engine approach to scaling?

**Answer:**

Cerebras takes the extreme approach of building the entire accelerator on a single silicon wafer (approximately 46,225 mm^2 for the WSE-2), eliminating chip-to-chip boundaries entirely. Key aspects:

**Monolithic fabric**: 850,000 cores connected by a 2D mesh with single-cycle nearest-neighbor latency. There are no inter-chip links, interposers, or package boundaries. Communication between any two cores uses only on-chip wires, providing orders of magnitude more bandwidth than chip-to-chip interconnects.

**Distributed SRAM**: 40 GB of SRAM distributed across cores (48 KB per core). This eliminates the HBM bandwidth bottleneck for models that fit in 40 GB. The aggregate on-chip bandwidth exceeds 20 PB/s.

**Yield management**: A wafer-scale chip cannot simply discard defective regions. Cerebras uses redundant cores and interconnect links to route around defects. The mesh topology is naturally fault-tolerant: if a core is defective, its neighbors can route traffic around it.

**Power delivery**: Distributing 15 kW uniformly across a 300mm wafer requires innovative power delivery. Cerebras uses a custom power distribution network with thousands of power pins distributed across the wafer.

**Weight streaming**: For models larger than 40 GB, Cerebras uses a Weight Streaming architecture where weights are stored in external memory servers (MemoryX) and streamed to the WSE over high-bandwidth links. Only activations reside on the WSE, and the massive on-chip bandwidth eliminates the memory wall for activation-related data movement.

**Scaling beyond one wafer**: Cerebras connects multiple WSE systems through a cluster fabric (SwarmX) for distributed training. This introduces conventional inter-system communication bottlenecks, but the within-system performance is unmatched.

---

### Q5. What are chiplets and how do they apply to AI accelerators?

**Answer:**

Chiplets are small, modular die that are combined in a single package to create a larger system. Instead of fabricating one large monolithic die, multiple smaller dies are manufactured separately and assembled using advanced packaging technology.

Advantages for AI accelerators: (1) better yield -- smaller dies have exponentially higher yield than larger dies, reducing cost; (2) heterogeneous integration -- different chiplets can use different process nodes (e.g., 3nm for compute, 5nm for I/O, old process for analog); (3) modular design -- the same compute chiplet can be combined in different configurations for different products; (4) exceeding reticle limits -- total silicon area can exceed the ~800 mm^2 maximum for a single die.

**AMD MI300X**: Uses 4 compute dies (CDNA 3 architecture on 5nm) plus 4 I/O dies (on 6nm) plus 8 HBM3 stacks, all on a single package. The compute chiplets are connected via an Infinity Fabric interconnect through the I/O dies. Total package provides 192 GB HBM3 at 5.3 TB/s.

**NVIDIA B200**: Uses two compute dies connected by a high-bandwidth on-package interconnect, creating a single logical GPU. The dual-die design allows the total transistor count and compute throughput to exceed what a single die could achieve at the given process node.

**Intel Gaudi 3**: Uses two compute dies connected via an internal bridge.

The key challenge for chiplets in AI accelerators is the inter-chiplet bandwidth. On-chip wire bandwidth density is approximately 1 TB/s/mm of edge, while advanced packaging provides approximately 100-200 GB/s/mm of edge. This 5-10x gap means that chiplet boundaries create bandwidth bottlenecks, and the architecture must minimize cross-chiplet traffic through careful data partitioning and scheduling.

---

### Q6. What are the key collective communication primitives and how are they implemented?

**Answer:**

**All-reduce**: Every node contributes a value (or vector), and every node receives the sum (or other reduction) of all contributions. Used for gradient synchronization in data-parallel training. Implemented as reduce-scatter (each node gets 1/N of the reduced result) followed by all-gather (each node broadcasts its portion to all others).

**All-gather**: Every node contributes a piece of data, and every node receives the concatenation of all pieces. Used in ZeRO-3 to reconstruct full parameters from distributed shards. Communication volume: (N-1)/N * total_data per node.

**Reduce-scatter**: Every node contributes a full vector, and each node receives 1/N of the reduced (summed) result. Communication volume: (N-1)/N * total_data per node.

**All-to-all**: Each node sends different data to each other node. Used in expert parallelism for routing tokens to experts.

**Broadcast**: One node sends the same data to all other nodes. Used for distributing model parameters or hyperparameters.

Implementation strategies:

**Ring algorithm**: Nodes are arranged in a logical ring. Data is divided into N chunks. In N-1 steps, each node sends one chunk to its neighbor and receives one chunk. For all-reduce: first N-1 steps perform reduce-scatter, next N-1 steps perform all-gather. Total communication: 2*(N-1)/N * data_size per node. Bandwidth-optimal but latency is O(N).

**Tree algorithm**: Uses a binary tree structure. Reduction happens in log(N) steps by combining pairs. Latency-optimal (O(log N)) but bandwidth-suboptimal for large data.

**Recursive halving-doubling**: Combines the latency advantage of tree algorithms with good bandwidth efficiency for medium-sized data.

**Hardware implementation**: NVIDIA's NCCL library implements these primitives on GPU clusters, using NVLink within nodes and InfiniBand between nodes. SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) offloads reduction to InfiniBand switches. NVLink Switch includes in-network reduction for NVLink domains.

---

### Q7. How does scale-out networking work for AI training clusters?

**Answer:**

Scale-out networking connects multiple server nodes (each containing 4-8 GPUs) into a training cluster. The network must provide sufficient bandwidth for collective communication (primarily all-reduce for data-parallel training and activation exchange for pipeline/tensor parallelism).

**Fat-tree topology**: The dominant data center network topology. Servers connect to leaf switches, leaf switches to spine switches, and spine switches to core switches (in a 3-tier design). A full bisection bandwidth fat tree provides non-blocking any-to-any communication, but this is expensive. Most practical deployments use a 2:1 or 3:1 oversubscription ratio (the upper tiers have less aggregate bandwidth than the lower tiers).

**Rail-optimized topology**: For GPU clusters where each node has 8 GPUs with 8 InfiniBand ports, each GPU connects to a different "rail" (leaf switch). GPUs at the same position in different nodes (e.g., GPU 0 in node 0 and GPU 0 in node 1) are on the same rail. This topology optimizes for the common all-reduce pattern where GPUs at the same position communicate.

**Dragonfly topology**: Uses groups of nodes connected by a local network, with global links between groups. Provides near-full-bisection bandwidth at lower switch count than a fat tree, but requires more complex routing.

**Bandwidth requirements**: For data-parallel training of a 70B parameter model with BF16 gradients: the all-reduce volume per step is approximately 140 GB. With 1024 GPUs in a ring all-reduce across 128 nodes, each node needs approximately 2 * 140 / 128 = 2.19 GB of inter-node communication per step. At 100 GB/s inter-node bandwidth, this takes 21.9 ms. If the compute per step is 500 ms, the communication overhead is approximately 4.4%, which is acceptable.

---

### Q8. What is the role of RDMA in AI training networks?

**Answer:**

RDMA (Remote Direct Memory Access) allows one node's NIC to directly read from or write to another node's memory without involving either node's CPU. This is critical for AI training because GPU-to-GPU communication performance depends on avoiding CPU overhead.

**Without RDMA**: A GPU sends data to the CPU via PCIe, the CPU copies it to the NIC buffer, the NIC transmits it over the network, the remote NIC receives it and copies to CPU memory, and the CPU copies it to the remote GPU via PCIe. This involves multiple memory copies and CPU involvement, adding latency and consuming CPU resources.

**With GPUDirect RDMA**: The NIC directly reads from and writes to GPU memory over PCIe. The communication path is GPU -> PCIe -> NIC -> Network -> NIC -> PCIe -> GPU, with no CPU involvement. This reduces latency by approximately 5-10 microseconds and frees the CPU for other tasks (data preprocessing, scheduling).

RDMA protocols used in AI clusters: InfiniBand Verbs (native RDMA on InfiniBand networks), RoCE v2 (RDMA over Converged Ethernet, using UDP/IP for transport), and iWARP (RDMA over TCP). InfiniBand Verbs provides the lowest latency and highest throughput, and is the dominant choice for high-performance AI training clusters. RoCE v2 is gaining adoption as Ethernet switches improve in performance and cost efficiency.

NVIDIA's NCCL library abstracts the RDMA transport layer, providing collective communication primitives that automatically use GPUDirect RDMA when available. The programmer writes high-level all-reduce calls; NCCL handles the low-level RDMA operations.

---

### Q9. What bandwidth is needed for different parallelism strategies at different scales?

**Answer:**

The bandwidth requirement depends on the parallelism strategy, model size, and number of accelerators:

**Data parallelism (all-reduce gradients)**:
- Communication per step: 2 * param_size * (N-1)/N (ring all-reduce)
- For 70B model, BF16: ~280 GB per step (for large N)
- Per-node bandwidth need: 280 / num_nodes GB per step
- Overlappable with backward pass computation

**Tensor parallelism (all-reduce activations)**:
- Communication per layer: 2 * batch * seq_len * model_dim * 2 bytes (forward + backward)
- For B=1, S=2048, D=8192: 2 * 2048 * 8192 * 2 = 64 MB per layer
- With 80 layers: 5.12 GB per step
- NOT overlappable (must complete before next layer)
- Requires very high bandwidth (NVLink-class)

**Pipeline parallelism (activation exchange)**:
- Communication per micro-batch: batch * seq_len * model_dim * 2 bytes
- Lower bandwidth than TP, but adds pipeline bubble overhead
- Latency-sensitive (pipeline bubbles increase with stage latency)

| Strategy | Comm volume (70B model) | Overlap? | Bandwidth class |
|---|---|---|---|
| DP (ring all-reduce) | ~280 GB/step | Yes (with backward) | InfiniBand (50-100 GB/s) |
| TP (per-layer all-reduce) | ~5 GB/step (B=1, S=2K) | No | NVLink (900 GB/s) |
| PP (activation passing) | ~64 MB/micro-batch | Partial | InfiniBand sufficient |

This explains the standard practice: TP within NVLink domains (8 GPUs), PP across nodes, DP across pipeline groups. Each strategy is matched to the bandwidth tier it requires.

---

### Q10. What are the emerging interconnect technologies for future AI systems?

**Answer:**

**UALink (Ultra Accelerator Link)**: An industry consortium (AMD, Broadcom, Cisco, Google, Intel, Meta, Microsoft) developing an open standard for GPU/accelerator-to-accelerator interconnect, as an alternative to NVIDIA's proprietary NVLink. UALink aims to provide NVLink-class bandwidth with an open specification, enabling multi-vendor interoperability. Version 1.0 targets up to 200 GB/s per link.

**CXL (Compute Express Link)**: A cache-coherent interconnect built on PCIe physical layer. CXL 3.0 enables memory pooling (shared memory across multiple hosts), peer-to-peer communication, and fabric-attached memory. For AI, CXL could enable disaggregated memory architectures where accelerators access shared memory pools for model weights, increasing effective memory capacity beyond local HBM.

**Photonic interconnects**: Silicon photonics promises much higher bandwidth-distance product than electrical signaling at lower power. Companies like Lightmatter (Passage), Ayar Labs, and Celestial AI are developing photonic chip-to-chip and chip-to-memory interconnects. Potential benefits: 10x bandwidth per fiber vs electrical, 10x lower power per bit, and longer reach without signal degradation.

**Co-packaged optics**: Integrating optical transceivers directly into the accelerator package (rather than on separate modules connected by electrical traces). This reduces the electrical signaling distance and enables higher bandwidth at the package boundary. NVIDIA, Broadcom, and others are investing in co-packaged optics for future data center networks.

**In-network compute**: Extending the SmartNIC and SHARP concepts to perform more computation within the network fabric itself (beyond simple reduction). This could enable pipelined all-reduce where the reduction is performed across network hops, reducing the effective latency of collective operations.
