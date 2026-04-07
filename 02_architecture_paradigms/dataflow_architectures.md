# Dataflow Architectures

This section covers spatial and dataflow architecture paradigms for AI accelerators, including the dataflow taxonomy (weight-stationary, output-stationary, row-stationary), and notable dataflow accelerator designs.

---

### Q1. What is a dataflow architecture and how does it differ from a von Neumann architecture?

**Answer:**

A dataflow architecture is one in which computation is driven by the availability of data rather than by a sequential program counter. In a von Neumann architecture (which includes CPUs and, to a large extent, GPUs), a central control unit fetches instructions from memory, decodes them, and dispatches them to execution units. The program counter determines execution order. In a dataflow architecture, processing elements (PEs) fire (execute their operation) as soon as all their input operands are available, with no central sequencer.

For AI accelerators, "dataflow architecture" typically refers to spatial architectures where:
1. The computation graph is mapped directly onto a physical array of PEs.
2. Data flows between PEs through fixed interconnections (wires or a NoC).
3. Each PE is configured to perform a specific operation (e.g., multiply, accumulate, activate).
4. Execution is determined by data dependencies, not a program counter.

The key advantages are: (a) minimal instruction fetch/decode overhead (each PE knows its operation at configuration time), (b) explicit data movement along known paths (reducing the energy cost of data transport), (c) natural pipeline parallelism (different stages of the computation execute simultaneously on different PEs), and (d) potential for very high energy efficiency since energy is spent on computation and short-distance data movement rather than on control logic and multi-level cache hierarchies.

The disadvantages are: (a) less flexibility (the PE array configuration must be changed for different computation graphs), (b) mapping complex, irregular computations onto a fixed spatial array is challenging, (c) PE utilization can be low if the computation graph does not fully utilize all PEs, and (d) the programming model is less mature than von Neumann approaches.

---

### Q2. Explain the weight-stationary, output-stationary, and row-stationary dataflow taxonomies.

**Answer:**

The dataflow taxonomy, formalized by Sze et al. in 2017, categorizes accelerator architectures by which data type is held stationary in the PEs to maximize reuse and minimize data movement. The three data types in a neural network computation are weights (W), input activations (I), and output activations / partial sums (O).

**Weight-stationary (WS)**: Each PE stores a weight value in its local register. Input activations are broadcast or streamed across PEs, and partial sums flow between PEs or to an accumulation buffer. The same weight is reused across all input activations that it multiplies (across batch elements and spatial positions in convolution). This minimizes weight memory accesses and is ideal when weights are reused many times (high batch size, large spatial dimensions). Example: Google TPU v1.

**Output-stationary (OS)**: Each PE is assigned to compute one output activation. The PE accumulates partial sums locally while receiving streamed weights and input activations. This minimizes partial sum movement and avoids the need for a separate accumulation buffer. It is advantageous when the inner dimension (reduction dimension K) is large, generating many partial sums per output. Example: ShiDianNao.

**Row-stationary (RS)**: Introduced by the Eyeriss architecture. Each PE is assigned one row of the 1D convolution (one row of the filter applied to one row of the input). The PE stores the relevant filter weights, receives input activations, and accumulates partial sums locally. Data reuse is maximized for all three data types simultaneously by exploiting convolutional reuse, filter reuse, and ifmap (input feature map) reuse at the PE level. This is particularly efficient for convolutional layers where all three types of reuse are present.

**No Local Reuse (NLR)**: A degenerate case where PEs have no local storage and all data is accessed from a shared global buffer. This approach relies entirely on the global buffer's bandwidth and is energy-inefficient for data movement. It is included in the taxonomy for completeness.

The optimal dataflow depends on the workload: WS is best for inference with high batch sizes, OS for GEMMs with large K dimensions, and RS for convolutions with spatial reuse. Modern accelerators often support multiple dataflows or use reconfigurable interconnects to adapt to different layer types.

---

### Q3. What is the Eyeriss architecture and what innovations did it introduce?

**Answer:**

Eyeriss, developed by researchers at MIT (Sze, Chen, Emer, et al.), is a pioneering spatial dataflow accelerator for CNNs that introduced several influential concepts:

**Row-stationary dataflow**: As described above, Eyeriss maximizes data reuse for all three data types (weights, input activations, partial sums) by mapping 1D convolution rows onto PEs. This provides better energy efficiency than weight-stationary or output-stationary approaches for typical CNN layers.

**Spatial array with local communication**: Eyeriss uses a 168-PE spatial array where PEs communicate with their neighbors via a network-on-chip. Each PE has a small local scratchpad (SRAM) for storing its assigned weights and partial sums. Data flows between PEs using configurable multicast/unicast patterns.

**Hierarchical memory**: The memory hierarchy includes PE-local scratchpads (0.5 KB each), a shared global buffer (108 KB), and off-chip DRAM. The dataflow is designed to minimize accesses to each successively more expensive level.

**Data compression**: Eyeriss exploits zero-value activations (common after ReLU) by using run-length encoding to skip zero operands, reducing both memory bandwidth and computation. The hardware detects zeros and gates off the multiplier, saving energy.

**Reconfigurable mapping**: The control logic can configure different PE-to-computation mappings for different layer types and dimensions, improving utilization across diverse CNN architectures.

Eyeriss demonstrated 10x better energy efficiency than a mobile GPU (2016) on CNN inference. Its contributions influenced many subsequent accelerator designs and established a framework (the dataflow taxonomy and energy model) that became standard tools for reasoning about accelerator efficiency.

Eyeriss v2 extended the architecture with a hierarchical mesh NoC for more flexible data delivery patterns, supporting a broader range of layer shapes and data reuse patterns.

---

### Q4. What is Cerebras Wafer-Scale Engine and how does it use dataflow principles?

**Answer:**

The Cerebras Wafer-Scale Engine (WSE) is the largest chip ever built, occupying an entire 300mm silicon wafer. The WSE-2 contains 2.6 trillion transistors, 850,000 AI-optimized cores, 40 GB of on-chip SRAM, and 220 petabits/s of on-chip interconnect bandwidth, consuming approximately 15 kW.

The architecture is fundamentally a spatial dataflow design at massive scale:

**Core array**: 850,000 small, energy-efficient cores are arranged in a 2D mesh. Each core has its own local SRAM (48 KB), a compute unit (FMAC for floating-point multiply-accumulate), and router connections to its 4 neighbors. There is no off-chip memory (no HBM or DRAM) -- all data resides in the distributed on-chip SRAM.

**Dataflow execution**: A neural network layer is mapped spatially across the core array. Data flows between cores following the computation graph. Weight matrices are distributed across the local SRAMs of many cores. Activations flow from core to core as they pass through the network layers.

**Advantages**: (a) Eliminates the memory wall -- 40 GB of on-chip SRAM at aggregate bandwidth exceeding 20 PB/s dwarfs any HBM-based design. (b) Massive parallelism -- 850,000 cores provide enormous throughput. (c) Low-latency communication -- core-to-core communication takes a single cycle (nearest neighbor).

**Challenges**: (a) Manufacturing yield -- a wafer-scale chip cannot discard defective dies as in traditional manufacturing, so Cerebras uses redundant cores and interconnect to route around defects. (b) Power delivery -- distributing 15 kW uniformly across a wafer requires innovative power delivery. (c) Cooling -- the entire wafer must be cooled, requiring custom cold-plate solutions. (d) Programming model -- mapping arbitrary models onto 850,000 cores requires sophisticated compiler technology. (e) Limited memory -- 40 GB on-chip SRAM limits the model size that can fit on a single WSE.

Cerebras has demonstrated strong performance on training and inference workloads, particularly for models that fit within the on-chip SRAM capacity. For larger models, their Weight Streaming architecture offloads weight storage to external memory servers while keeping activations on the WSE.

---

### Q5. How does Graphcore's Intelligence Processing Unit (IPU) differ from GPUs and TPUs?

**Answer:**

Graphcore's IPU uses a Bulk Synchronous Parallel (BSP) execution model with a large distributed SRAM and exchange fabric, representing a distinct point in the design space:

**Architecture**: Each IPU contains 1,472 independent processing tiles (in the Mk2 Colossus GC200). Each tile is a fully programmable processor with its own local SRAM (624 KB per tile, 897 MB total per chip), 6 hardware thread contexts, and a compute unit supporting FP32, FP16, and other formats.

**BSP execution model**: Computation proceeds in alternating compute and exchange phases. During the compute phase, all tiles execute independently on their local data. During the exchange phase, tiles communicate via an all-to-all exchange fabric (8 TB/s on-chip bandwidth). This is fundamentally different from GPUs (which use a shared memory hierarchy) and TPUs (which use streaming dataflow).

**No off-chip memory**: Like Cerebras, the IPU relies entirely on distributed on-chip SRAM. There is no HBM. This eliminates the memory wall for models that fit, but limits the maximum model size per chip. Graphcore addresses larger models through multi-IPU configurations (IPU-POD systems with up to 256 IPUs).

**Comparison with GPUs**: IPUs have higher aggregate on-chip memory bandwidth (8 TB/s vs 3.35 TB/s HBM on H100) but much less total memory capacity (897 MB vs 80 GB). This makes IPUs potentially advantageous for memory-bound operations on small models but disadvantageous for large models with large weight matrices.

**Comparison with TPUs**: IPUs are more programmable (general-purpose tile processors vs fixed-function systolic arrays) but less energy-efficient for pure matrix multiplication. The BSP model provides clear synchronization semantics but can be inefficient when compute load is unbalanced across tiles.

Graphcore's Poplar software stack includes a graph compiler that maps neural network computations onto tiles, handles data partitioning, and generates exchange schedules. The IPU architecture has found niches in applications requiring fine-grained parallelism and irregular computation patterns, though it has struggled to compete with NVIDIA GPUs for mainstream transformer training.

---

### Q6. What is Groq's LPU architecture and how does it achieve low latency?

**Answer:**

Groq's Language Processing Unit (LPU) is a deterministic dataflow architecture designed specifically for low-latency inference on large language models. Its key innovation is a compiler-scheduled architecture that eliminates runtime scheduling overhead entirely.

**Temporal instruction set architecture (TISA)**: Unlike GPUs where hardware schedulers dynamically assign work, the Groq compiler statically schedules every operation, every data movement, and every memory access at compile time. The resulting schedule is a fixed sequence of instructions where every cycle's activity across the entire chip is predetermined. This eliminates the need for caches (no cache misses, since memory accesses are scheduled), branch predictors, and dynamic scheduling hardware.

**Architecture**: The LPU chip contains a large SRAM (230 MB on the GroqChip 1) organized as a distributed memory across many functional units connected by a high-bandwidth on-chip network. Compute units include vector processing elements and matrix multiply units. The functional units and memory are organized in a 2D layout with deterministic, compiler-scheduled data movement.

**Low latency**: By eliminating all sources of nondeterminism (cache misses, dynamic scheduling, memory access conflicts), every inference request takes exactly the same number of cycles. This provides extremely consistent and low latency, which is valuable for real-time inference applications. Groq has demonstrated sub-millisecond first-token latency on large language models.

**Tradeoffs**: (a) The compiler must produce a complete static schedule, which is computationally expensive and limits model size to what fits in on-chip SRAM. (b) Dynamic shapes (variable sequence lengths, dynamic batching) are more difficult to handle in a statically scheduled architecture. (c) The design prioritizes latency over throughput, making it less cost-efficient for throughput-oriented batch inference.

Groq's approach represents a bet that for many inference deployments, latency and latency consistency matter more than peak throughput, and that compiler technology can substitute for hardware scheduling complexity.

---

### Q7. How does SambaNova's Reconfigurable Dataflow Unit (RDU) work?

**Answer:**

SambaNova's RDU is a reconfigurable dataflow architecture that aims to combine the efficiency of fixed-function accelerators with the flexibility of programmable processors. The architecture uses a hierarchy of reconfigurable compute and memory units connected by a pattern-based data flow network.

**Pattern Compute Units (PCUs)**: Configurable SIMD compute elements that can be programmed to execute different arithmetic operations (multiply-accumulate, activation functions, normalization). Each PCU contains multiple parallel lanes and stages that can be configured as a pipeline for different computation patterns.

**Pattern Memory Units (PMUs)**: Configurable on-chip SRAM banks with address generators that can produce various access patterns (strided, tiled, transposed) without consuming compute cycles for address calculation. PMUs serve as scratchpads, FIFOs, or lookup tables depending on the configuration.

**Interconnect**: A dedicated network connects PCUs and PMUs with configurable routing. Data flows between units according to a dataflow graph produced by the compiler. The network supports multicast (one-to-many) and gather (many-to-one) patterns.

**Reconfigurability**: The key differentiator from fixed-function systolic arrays is that PCUs and PMUs can be reconfigured for different computation patterns. A PCU might be configured as a MAC array for matrix multiplication in one layer and as an element-wise operation pipeline for activation functions in the next. This reconfiguration happens between layers of the neural network, allowing a single hardware substrate to efficiently execute diverse operation types.

**Software**: SambaNova's compiler maps the neural network computation graph onto the RDU's physical resources, determining which PCUs execute which operations, how data is partitioned across PMUs, and how data flows through the interconnect. The compiler performs tiling, scheduling, and resource allocation automatically.

The RDU approach attempts to capture the energy efficiency of dataflow execution (data moves along wires between units rather than through a cache hierarchy) while providing more flexibility than a systolic array. It is well-suited to applications requiring diverse computation patterns, such as enterprise AI workloads that involve not just matrix multiplication but also data preprocessing, search, and post-processing.

---

### Q8. What is the difference between temporal and spatial architectures?

**Answer:**

This distinction, articulated by Sze et al. and Patterson/Hennessy, is fundamental to understanding AI accelerator design:

**Temporal architecture**: A single or small number of complex processing elements execute a sequence of operations over time, reading operands from and writing results to a shared memory hierarchy. CPUs and GPUs are temporal architectures. An operation at time T uses the same ALU as an operation at time T+1 -- the computation unfolds temporally. Data reuse is achieved through caches (implicit) or scratchpads (explicit).

**Spatial architecture**: Many simple processing elements are arranged in space, each performing a specific operation. Data flows between PEs through direct connections. The computation graph is physically laid out across the PE array -- different operations happen simultaneously in different physical locations. Data reuse is achieved through dataflow: an operand is computed by one PE and consumed by a neighboring PE without ever going to a shared memory.

| Aspect | Temporal | Spatial |
|---|---|---|
| Execution model | Time-multiplexed on shared ALUs | Spatially distributed across PEs |
| Data movement | Through memory hierarchy (caches, registers) | Along wires between PEs |
| Control | Centralized (instruction stream) | Distributed (local PE control or compiler-scheduled) |
| Energy profile | Dominated by memory hierarchy | Dominated by computation + local wiring |
| Flexibility | High (any instruction stream) | Lower (fixed or reconfigurable PE functions) |
| Examples | CPU, GPU | Systolic array, Eyeriss, Cerebras WSE |

In practice, most modern AI accelerators are hybrids. A GPU is primarily temporal (warp schedulers execute instruction streams on shared execution units) but its tensor cores have spatial characteristics (the MAC array within a tensor core is a small spatial dataflow pipeline). A TPU is primarily spatial (the systolic array) but has temporal elements (the vector unit for non-GEMM operations).

---

### Q9. How do dataflow architectures handle non-GEMM operations like softmax and normalization?

**Answer:**

Non-GEMM operations (softmax, layer normalization, activation functions, residual connections) present a challenge for dataflow architectures designed around matrix multiplication, because these operations have different computation patterns:

**Dedicated functional units**: Many accelerators include separate functional units alongside the matrix multiply array. The TPU has a vector processing unit adjacent to the MXU that handles activation functions, normalization, and other element-wise operations. After the MXU computes a matrix multiply output, the result passes through the vector unit before being written back to memory or fed into the next MXU operation.

**Reconfigurable PEs**: In architectures like SambaNova's RDU, the same PEs that perform matrix multiplication can be reconfigured for other operations between computation phases. This avoids dedicated hardware for each operation type at the cost of reconfiguration overhead.

**Software fusion**: The compiler fuses multiple non-GEMM operations into a single pass over the data. For example, bias addition, activation function, and residual connection can be fused into one read-compute-write pass, reducing memory bandwidth requirements.

**Specialized hardware for attention**: Given the importance of transformer attention, some architectures include dedicated hardware for softmax computation (requiring exponentiation, reduction, and division) and for attention score masking.

**On-chip buffers**: Non-GEMM operations are typically memory-bound, so keeping intermediate results in on-chip SRAM (rather than writing to HBM and reading back) is critical. Double-buffering schemes allow the matrix multiply unit to compute the next tile while the vector unit processes the current tile's non-GEMM operations.

The overall design principle is that non-GEMM operations should not become the bottleneck even though they are a small fraction of total FLOPS. This requires sufficient vector processing capability and memory bandwidth for element-wise operations, plus compiler support for operation fusion.

---

### Q10. What is the role of the compiler in a dataflow accelerator?

**Answer:**

The compiler is arguably more critical for dataflow accelerators than for GPUs, because the hardware provides less dynamic flexibility and the compiler must make decisions that GPUs handle in hardware:

**Graph partitioning and mapping**: The compiler must partition the computation graph across the physical PE array, assigning operations to specific PEs and determining which PEs communicate. For a systolic array, this means deciding how to tile matrices. For a spatial dataflow architecture like Cerebras WSE, this means distributing layers across 850,000 cores.

**Scheduling**: In a statically scheduled architecture (like Groq), the compiler determines the exact cycle at which each operation fires and each data transfer occurs. In a dynamically scheduled architecture, the compiler determines operation order within each PE. Scheduling quality directly impacts utilization and latency.

**Memory allocation**: The compiler assigns data to specific memory locations (which PE's local SRAM, which bank of a shared buffer) and generates address patterns for data access. For architectures without caches, this is the only mechanism for data placement.

**Data layout optimization**: The compiler chooses how matrices are stored in memory (row-major, column-major, tiled, interleaved) to match the hardware's access patterns. A poor layout choice can cause bank conflicts or underutilize the interconnect.

**Tiling and loop ordering**: For operations too large to fit on the PE array in one pass, the compiler determines the tiling strategy (tile sizes, loop ordering) to maximize data reuse and minimize off-chip memory accesses.

**Communication scheduling**: For multi-chip systems, the compiler schedules inter-chip communication (all-reduce, activation transfers) to overlap with computation and minimize idle time.

The compiler's quality can make or break a dataflow accelerator. A good compiler achieves high PE utilization and efficient data movement; a poor compiler leaves PEs idle and wastes memory bandwidth. This is why companies like Groq, Cerebras, SambaNova, and Google invest heavily in compiler engineering alongside hardware design.
