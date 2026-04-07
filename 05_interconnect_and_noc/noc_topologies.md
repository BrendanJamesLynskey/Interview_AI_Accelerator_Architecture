# NoC Topologies

This section covers network-on-chip topologies used in AI accelerators, including mesh, ring, tree, and crossbar designs, as well as bandwidth provisioning and flow control.

---

### Q1. What are the key metrics for evaluating NoC topologies?

**Answer:**

**Bisection bandwidth**: The minimum bandwidth across any cut that divides the network into two equal halves. This represents the worst-case all-to-all communication capacity and is the most important scalability metric. A mesh of N nodes with link bandwidth B has bisection bandwidth of sqrt(N)*B. A crossbar has bisection bandwidth of N*B/2.

**Diameter**: The maximum number of hops between any pair of nodes. Lower diameter means lower worst-case latency. Mesh: 2*(sqrt(N)-1). Ring: N/2. Crossbar: 1. Tree: 2*log(N).

**Average hop count**: The expected number of hops for a random source-destination pair. Mesh: (2/3)*sqrt(N). This affects average latency and power (each hop consumes router energy).

**Degree**: The number of links per node. Higher degree means more hardware per node but potentially lower hop count. Mesh: 4 (interior nodes). Ring: 2. Tree: 3 (interior). Crossbar: N.

**Cost (area and power)**: Router area scales with degree (crossbar within router) and buffer size. Link area scales with wire width and length. Total network cost includes all routers, links, and buffers.

**Throughput under contention**: How well the network handles hot-spot traffic patterns (many nodes sending to the same destination) or adversarial patterns. This depends on routing algorithm, flow control, and topology.

For AI accelerators, the key tradeoff is between bisection bandwidth (which determines collective communication performance) and cost (area and power). The optimal topology depends on the size of the network and the dominant communication patterns.

---

### Q2. How does a 2D mesh topology work and why is it popular for AI accelerators?

**Answer:**

A 2D mesh arranges N nodes in a sqrt(N) x sqrt(N) grid. Each interior node connects to its 4 nearest neighbors (north, south, east, west). Edge nodes have 2-3 connections. Communication between non-adjacent nodes is routed through intermediate nodes using algorithms like XY routing (go east/west first, then north/south) or adaptive routing.

The 2D mesh is popular for AI accelerators because it maps naturally onto a 2D silicon die. The nodes (compute tiles, memory banks, or PE clusters) are physically arranged in a grid, and the mesh wires follow the chip's metal layers. This physical correspondence means that wire lengths are short (only to nearest neighbors) and routing is straightforward.

For AI workloads, the 2D mesh supports the common communication patterns efficiently. Systolic data flow (streaming data along rows or columns) maps directly to mesh rows and columns. Multicast along a row or column requires only linear-time flooding. Reduction along a row or column is also naturally supported.

The mesh's bisection bandwidth scales as sqrt(N) * link_bandwidth, which is sub-linear in N. For very large networks, this can become a bottleneck for all-to-all communication patterns (like all-reduce). This limitation motivates alternative topologies (torus, fat tree) for large-scale systems or the use of dedicated global reduction networks alongside the local mesh.

Cerebras WSE uses a 2D mesh connecting 850,000 cores. Google TPU pods extend the mesh concept with wraparound links (creating a torus) for improved bisection bandwidth and reduced diameter.

---

### Q3. What advantages does a torus have over a mesh?

**Answer:**

A torus extends a mesh by adding wraparound links that connect the edge nodes to the opposite edge, forming a ring in each dimension. A 2D torus of size K x K has the same number of nodes and links as a mesh plus K additional links per dimension (2K total wraparound links).

**Advantages:**
1. **Reduced diameter**: Mesh diameter is 2*(K-1); torus diameter is K (half as many maximum hops in each dimension). This reduces worst-case latency by roughly 2x.
2. **Doubled bisection bandwidth**: The wraparound links provide additional paths across the bisection, doubling the bisection bandwidth compared to a mesh of the same size.
3. **Uniform bandwidth**: Every node in a torus has the same degree (4 for 2D), including edge nodes. In a mesh, edge and corner nodes have fewer connections, creating bandwidth asymmetry.
4. **Better load balancing**: The wraparound links distribute traffic more evenly, reducing hotspots at the center of the network (which is a common problem in meshes).

**Disadvantages:**
1. **Long wires**: Wraparound links span the entire chip dimension, which can be 10-20mm. These long wires have higher latency, higher power, and may require repeaters or pipelining.
2. **Routing complexity**: Deadlock-free routing in a torus requires more virtual channels or more complex routing algorithms than in a mesh (where simple XY routing is deadlock-free).
3. **Physical layout**: The wraparound wires cross over many other wires, complicating the physical layout and potentially requiring dedicated metal layers.

Google's TPU interconnect (ICI) uses a 3D torus, accepting the long-wire costs in exchange for the bandwidth and latency benefits. The 3D torus provides six links per node (two per dimension) and excellent all-reduce performance when decomposed into three independent ring all-reduces along the three dimensions.

---

### Q4. When is a crossbar appropriate for an on-chip interconnect?

**Answer:**

A crossbar provides non-blocking, single-hop connectivity between all input-output pairs. Any input can communicate with any output simultaneously at full link bandwidth, with no interference from other communications (assuming no output conflicts).

Crossbars are appropriate when: (1) the number of ports is small (up to ~32-64), (2) full bisection bandwidth is needed, and (3) latency must be minimal and deterministic. In AI accelerators, crossbars are commonly used for:

- SM-to-L2-cache interconnect in GPUs (connecting ~132 SMs to ~32 L2 partitions)
- Register file ports within a compute unit (connecting multiple functional units to register bank entries)
- Shared memory bank arbitration (connecting warp threads to memory banks)

The area of an NxN crossbar scales as O(N^2) because each input must have a wire path to each output. For 16-bit data width and N=32, the crossbar requires 32 * 32 * 16 = 16,384 crosspoints, each being a pass transistor or multiplexer bit. At 7nm, this is roughly 0.1-0.3 mm^2 -- acceptable for a small crossbar.

At N=256, the crossbar area grows to approximately 10 mm^2, which begins to compete with the compute units themselves. At N=1024 or higher, a crossbar is impractical, and multi-stage switching networks (Clos networks, butterfly networks) or mesh/torus topologies are used instead.

---

### Q5. How do hierarchical interconnects balance local and global communication?

**Answer:**

A hierarchical interconnect uses different topologies at different scales, optimizing each level for its characteristic traffic patterns and physical constraints.

A common hierarchy for AI accelerators:
1. **PE cluster level**: 4-16 PEs connected by a local crossbar or bus. Very low latency (1-2 cycles), very high bandwidth. Handles data sharing within a GEMM tile computation.
2. **Tile/SM level**: 8-32 PE clusters connected by a local mesh or ring. Moderate latency (5-10 cycles). Handles data sharing within a larger computation phase (e.g., different output tiles of a GEMM).
3. **Chip level**: Multiple tiles connected by a global mesh, ring, or tree. Higher latency (20-100 cycles). Handles data movement between tiles for different layers or different parts of a partitioned model.
4. **Multi-chip level**: Chips connected by NVLink, ICI, or InfiniBand. Much higher latency (1000+ ns). Handles gradient synchronization and activation exchange for distributed training.

The benefit of hierarchy is that most communication is local (within a tile or cluster), and local communication is fast and cheap. Only a small fraction of traffic needs to traverse the global network, so the global network can have lower bandwidth per node without becoming a bottleneck.

The design challenge is determining the right bandwidth at each level. If the global network is too narrow, collective operations (all-reduce, all-gather) become bottlenecks. If it is too wide, area is wasted. Traffic analysis of target workloads, combined with analytical models and simulation, guides the bandwidth provisioning at each level.

---

### Q6. What routing algorithms are used in AI accelerator NoCs?

**Answer:**

**Deterministic XY routing**: In a 2D mesh or torus, packets first travel along the X dimension, then along the Y dimension. This is simple, low-overhead, and deadlock-free (in a mesh; torus requires additional VCs). It produces predictable latency and is easy to implement in hardware. Used in most mesh-based AI accelerator NoCs.

**Dimension-ordered routing**: Generalization of XY routing to higher dimensions. In a 3D torus, packets travel along dimension 0, then dimension 1, then dimension 2. Used in TPU ICI.

**Adaptive routing**: Packets can choose among multiple paths based on network congestion. When the preferred path is congested, the packet takes an alternative route. This improves throughput under non-uniform traffic but adds routing complexity and can cause out-of-order packet delivery.

**Minimal routing**: Packets take only shortest paths (no detours). This minimizes hop count and latency but may create hotspots if many packets compete for the same shortest path.

**Source routing**: The complete route is determined at the source and encoded in the packet header. Each intermediate node simply reads the next hop from the header. This eliminates routing lookup logic at intermediate nodes but requires the source to know the network state.

**Circuit switching**: A dedicated path is established between source and destination before data transmission. The path remains reserved for the duration of the transfer. This eliminates per-packet routing overhead and guarantees bandwidth, but the path setup latency is high and unused bandwidth on the reserved path is wasted. Some dataflow accelerators use circuit-switched interconnects for their deterministic communication patterns.

For AI accelerators, deterministic routing is preferred because the communication patterns are predictable (known at compile time or determined by the tiling schedule). Adaptive routing adds hardware complexity with limited benefit for regular traffic patterns.

---

### Q7. What is the impact of NoC latency on AI accelerator performance?

**Answer:**

NoC latency impacts AI accelerator performance differently depending on the workload and architecture:

**Systolic arrays and dataflow architectures**: NoC latency primarily affects the startup time of each new tile computation (the time to load new data into the PE array). Once the pipeline is filled, data flows through the array at one element per cycle, and the NoC latency is hidden. For large tiles (high K dimension), the startup cost is amortized and NoC latency has minimal impact. For small tiles or frequent tile changes, NoC latency can significantly reduce utilization.

**GPU SMs**: NoC latency between SMs and L2/HBM affects memory access stalls. GPUs hide this latency through massive thread-level parallelism: while one warp stalls on a memory access, other warps execute. With sufficient occupancy (enough active warps), the NoC/memory latency is fully hidden, and the bottleneck is bandwidth rather than latency.

**Collective operations**: For all-reduce and other collective communications, the total latency includes the number of network hops times the per-hop latency. In a ring all-reduce on a mesh, the message traverses up to N-1 hops. If per-hop latency is 10 ns and N=64, the communication latency is at least 630 ns (plus serialization time for the data). This can be significant for small data volumes (e.g., partial all-reduce of a single layer's gradients).

**Inference latency**: For latency-critical inference (e.g., real-time serving), every component of latency matters. A 100 ns NoC latency per layer, across 80 layers, adds 8 microseconds to the total inference latency. While this is small compared to the compute time for large models, it becomes significant for small models where compute time per layer is also in the microsecond range.

In general, bandwidth is more important than latency for AI workloads due to their high parallelism and large data volumes. However, latency becomes important at small scale (small tiles, small batches, few layers) and for collective operations.

---

### Q8. How do virtual channels prevent deadlock in torus networks?

**Answer:**

Deadlock occurs in a network when a set of packets are each waiting for resources held by other packets in a circular dependency. In a torus network with XY routing, deadlock can occur because the wraparound links create cycles in the channel dependency graph.

Consider a 1D ring (torus) with packets routed in one direction. Packet A at node 0 wants to go to node 3, and packet B at node 3 wants to go to node 0. If packet A holds the link from node 0 to node 1 and waits for the link from node 2 to node 3 (held by another packet), while packet B holds the link from node 3 to node 0 and waits for a link held by packet A, deadlock occurs.

Virtual channels (VCs) break the circular dependency by providing multiple independent buffer queues per physical link. The key technique is dateline routing: one point on each ring dimension is designated as a "dateline." Packets that cross the dateline switch from VC0 to VC1. Since no packet can switch from VC1 back to VC0, the circular dependency is broken.

With 2 VCs per link in a 2D torus, deadlock is avoided. The overhead is doubling the buffer space per link and adding VC arbitration logic. For higher-dimensional tori, D+1 VCs per dimension (or 2 VCs with a more complex routing algorithm) are sufficient.

In practice, AI accelerator NoCs often use 2-4 VCs regardless of deadlock requirements, because additional VCs also improve throughput by reducing head-of-line blocking (a blocked packet in one VC does not block packets in other VCs on the same link).

---

### Q9. How is bandwidth allocated in a NoC with heterogeneous traffic?

**Answer:**

AI accelerators generate different types of traffic with different priorities and bandwidth requirements:

1. **Data (weight/activation) reads**: High bandwidth, latency-tolerant. These are large transfers that dominate total traffic volume.
2. **Partial sum/gradient writes**: High bandwidth, latency-tolerant. Similar to data reads in character.
3. **Control signals**: Low bandwidth, latency-sensitive. Thread scheduling, synchronization, and memory management messages.
4. **Collective communication**: Medium bandwidth, latency-sensitive. All-reduce, all-gather for distributed computation.

Bandwidth allocation mechanisms:
- **Priority-based arbitration**: Different traffic types are assigned priorities. Control signals (highest priority) preempt data transfers. Collective communication gets medium priority. Data transfers (lowest priority, most bandwidth) fill the remaining capacity.
- **Bandwidth reservation**: A fraction of each link's bandwidth is statically reserved for each traffic class. This guarantees minimum bandwidth for critical traffic but may waste bandwidth if a class has no traffic to send.
- **Weighted fair queuing**: Each traffic class has a weight that determines its share of link bandwidth. Under contention, bandwidth is divided proportionally to the weights.
- **Dedicated networks**: Some accelerators use physically separate networks for different traffic types. For example, a dedicated reduction tree for gradients alongside a general-purpose mesh for data transfers. This provides isolation but increases area.

The choice depends on the traffic mix predictability. AI workloads have highly predictable traffic patterns (the compiler knows exactly when and where data will be transferred), so static bandwidth allocation and priority-based arbitration work well.

---

### Q10. What is the relationship between NoC design and compiler scheduling?

**Answer:**

In AI accelerators, especially dataflow and statically scheduled architectures, the compiler and NoC are tightly co-designed. The compiler generates a schedule that specifies exactly when each data transfer occurs and which network path it takes. The NoC must support these scheduled transfers without conflicts.

**Compiler responsibilities**: The compiler performs traffic scheduling: determining the cycle-accurate timing of each data transfer to avoid link conflicts. It performs route computation: choosing paths through the NoC for each transfer. It performs buffer management: allocating NoC buffer space for in-flight data.

**NoC requirements from the compiler**: Deterministic latency (so the compiler can predict when data arrives), sufficient bandwidth on each link for the scheduled traffic, and deadlock freedom under the compiler-generated traffic pattern.

**Co-design benefits**: When the compiler controls the schedule, the NoC can be simplified. Credit-based flow control may be replaced by static scheduling (the compiler guarantees no buffer overflow). Routing tables can be replaced by hardcoded routes (the compiler embeds the route in the packet header). Arbitration logic can be simplified because the compiler ensures no conflicts.

**Example**: In Groq's LPU architecture, the compiler produces a complete cycle-by-cycle schedule for all compute, memory, and communication operations. The on-chip network executes this schedule deterministically with no runtime arbitration. This eliminates the hardware complexity of dynamic flow control and arbitration, saving area and power.

For more flexible architectures (like GPUs), the NoC must handle dynamic, unpredictable traffic patterns. Here, the NoC design is more independent of the compiler, and hardware arbitration and flow control are essential. The compiler's role is limited to optimizing data locality (placing communicating thread blocks on nearby SMs) rather than scheduling individual transfers.
