# Systolic Arrays and TPUs

This section covers the architecture of systolic arrays and Google's Tensor Processing Units (TPUs), including matrix multiply unit design, bfloat16, and the evolution from TPU v1 through v5.

---

### Q1. What is a systolic array and how does it perform matrix multiplication?

**Answer:**

A systolic array is a grid of processing elements (PEs) arranged in a regular pattern (typically 2D), where data flows rhythmically through the array like blood pulses through the circulatory system (hence "systolic," from the Greek word for heart contraction). Each PE performs a simple operation (typically multiply-accumulate) and passes data to its neighbors.

For matrix multiplication C = A * B on an N x N systolic array, the operation proceeds as follows: Elements of matrix A flow from left to right across each row of PEs. Elements of matrix B flow from top to bottom down each column. Each PE multiplies the A element passing through it with the B element passing through it, accumulates the result into a local register, and passes both inputs onward to the next PE.

The key insight is that after an initial filling phase (2N-1 cycles), every PE is performing a useful multiply-accumulate every cycle. For an N x N array, this delivers N^2 MACs per cycle at steady state, providing extremely high compute density with minimal control overhead. There is no instruction fetch, no branch prediction, no register renaming -- just a fixed dataflow pattern with hardwired interconnections.

The initial fill and final drain phases create a startup cost of 2N-1 cycles. For a full matrix multiplication of dimension N on an N x N array, the total execution time is 3N-2 cycles, achieving a utilization of N^3 / (N^2 * (3N-2)) = N / (3N-2), which approaches 33% for large N. This is improved by tiling larger matrices into blocks that are streamed through the array, allowing the array to remain in steady state for longer periods.

---

### Q2. What are the different dataflow strategies in systolic arrays?

**Answer:**

The dataflow strategy determines which operand is stationary (stored in PE registers) and which operands flow through the array. The three primary strategies are:

**Weight-stationary (WS)**: Weights are preloaded into PE registers and remain stationary. Input activations flow through the array, and partial sums are accumulated locally. This minimizes weight data movement, which is advantageous when the same weights are reused across many input samples (high batch size). Google's TPU v1 uses a weight-stationary design.

**Output-stationary (OS)**: Partial sums for each output element remain in a fixed PE. Input activations and weights both flow through the array. This minimizes partial sum movement and avoids the need for a large accumulation buffer. It is advantageous when output elements require many partial sums (large inner dimension K).

**Row-stationary (RS)**: A more flexible approach introduced by the Eyeriss architecture. Each PE is assigned a row of the convolution operation, and it keeps a partial sum stationary while receiving both weights and input activations. This maximizes data reuse for all three data types (inputs, weights, partial sums) and is particularly efficient for convolutional layers with various filter sizes.

The choice of dataflow affects energy consumption and performance:

| Dataflow | Weight movement | Input movement | Partial sum movement |
|---|---|---|---|
| Weight-stationary | Minimal | High | Medium |
| Output-stationary | High | High | Minimal |
| Row-stationary | Medium | Medium | Low |

For modern AI accelerators targeting transformer workloads (where the dominant operation is large GEMMs rather than convolutions), weight-stationary and output-stationary designs are most common. The TPU's weight-stationary approach is particularly efficient for inference, where the same weight matrix is applied to many different inputs.

---

### Q3. Describe the architecture of Google's TPU v1 and its design philosophy.

**Answer:**

Google's TPU v1 (2015) was the first custom ASIC designed specifically for neural network inference. Its design philosophy prioritized inference throughput and efficiency over flexibility, and it represents one of the clearest examples of a domain-specific architecture.

The core compute unit is a 256 x 256 systolic array of 8-bit integer multiply-accumulate units, providing 65,536 MACs per cycle. At 700 MHz, this delivers approximately 92 TOPS of INT8 compute. The array uses a weight-stationary dataflow: weights are loaded into the array from an on-chip weight FIFO, and input activations flow through from left to right.

The memory system includes: 28 MB of on-chip SRAM (the "Unified Buffer") for storing activations, a 4 MB weight FIFO for staging weights before loading into the systolic array, and an 8 GB off-chip DRAM (DDR3) at 34 GB/s bandwidth.

Key design decisions:
- **INT8 only for multiply**: The TPU v1 was designed for inference of quantized models only, not training. INT8 operations are much cheaper than floating-point.
- **No general-purpose programmability**: The TPU v1 is a coprocessor controlled by the host CPU. It executes a small set of CISC-style instructions (read weights, multiply, activate, write results).
- **Large systolic array**: The 256x256 array prioritizes peak throughput over flexibility. Smaller matrix dimensions underutilize the array.
- **Modest memory bandwidth**: The 34 GB/s DRAM bandwidth reflects the design's focus on inference with high batch sizes, where weight reuse is high.

The TPU v1 achieved 15-30x better performance per watt than contemporary CPUs and GPUs on production inference workloads at Google, validating the domain-specific architecture approach. Its limitations (no training support, INT8 only, limited flexibility) motivated the more general TPU v2 and subsequent generations.

---

### Q4. How did the TPU evolve from v2 to v5, and what changed architecturally?

**Answer:**

**TPU v2 (2017)**: The first TPU to support training. It introduced bfloat16 (BF16) floating-point format, enabling training with sufficient dynamic range. The Matrix Multiply Unit (MXU) was 128x128 BF16 MAC units. Each chip had two cores, each with its own 128x128 MXU and 16 GB HBM. TPU v2 pods connected 256 chips via a 2D torus interconnect for distributed training.

**TPU v3 (2018)**: Doubled the compute per chip (2x FLOPS vs v2) by running the MXU at higher frequency and adding liquid cooling (the first liquid-cooled TPU). Memory increased to 32 GB HBM per chip. Pods scaled to 1024 chips. The 2D torus interconnect provided direct all-to-all communication without switches.

**TPU v4 (2021)**: A major architectural revision. Each chip had two MXUs per core, doubled to four MXUs total. The MXU supported BF16, FP32 accumulation, and INT8. 32 GB HBM2 per chip at higher bandwidth. The interconnect evolved from 2D torus to 3D torus (a 4x4x4 cube of 64 chips per pod unit), reducing the diameter of the network and improving all-reduce performance. TPU v4 pods scaled to 4096 chips.

**TPU v5e (2023)**: Optimized for inference efficiency. A cost-reduced design with a single core per chip, targeting high-volume inference serving. Lower TDP and lower cost per chip compared to v4, but optimized for throughput-per-dollar on inference workloads.

**TPU v5p (2023)**: The training-focused counterpart. Higher compute throughput, more HBM capacity (95 GB HBM2e), and higher interconnect bandwidth than v5e. Designed for large-scale training of foundation models.

The evolution shows several trends: transition from inference-only (v1) to training+inference (v2+), increasing MXU size and count per chip, growing HBM capacity and bandwidth, increasingly sophisticated interconnect topologies (2D torus to 3D torus), and the introduction of inference-optimized (v5e) and training-optimized (v5p) product lines.

---

### Q5. What is bfloat16 and why did Google develop it for the TPU?

**Answer:**

Bfloat16 (BF16 or Brain Float 16) is a 16-bit floating-point format with 1 sign bit, 8 exponent bits, and 7 mantissa bits. Google developed it specifically for deep learning training on TPUs, and it has since been widely adopted across the industry (NVIDIA, AMD, Intel, Apple, ARM).

The key insight behind BF16 is that the dynamic range (determined by exponent bits) matters more than precision (determined by mantissa bits) for neural network training. Standard FP16 (IEEE 754 half-precision) has 5 exponent bits and 10 mantissa bits. Its limited dynamic range (max value approximately 65,504) causes frequent overflows during training, requiring careful loss scaling. BF16's 8 exponent bits provide the same dynamic range as FP32 (max value approximately 3.4 * 10^38), eliminating most overflow concerns.

The tradeoff is reduced precision: BF16 has only 7 mantissa bits compared to FP16's 10, providing approximately 2-3 decimal digits of precision versus FP16's 3-4 digits. In practice, this reduced precision has minimal impact on training convergence for most models, because: (1) stochastic gradient descent is inherently noisy and tolerant of low-precision arithmetic, (2) critical accumulations (partial sums in matrix multiplications) are performed in FP32, and (3) the final model weights are typically stored in FP32 or higher precision.

BF16 is also convenient from a hardware and software perspective because converting between BF16 and FP32 requires only truncating or rounding the lower 16 bits of the mantissa, with no exponent adjustment needed. This makes BF16-FP32 conversion nearly free in hardware.

The adoption of BF16 across the industry validates Google's design choice and demonstrates how hardware design (the TPU's need for efficient training formats) can influence software standards (BF16 is now supported in PyTorch, TensorFlow, JAX, and most deep learning frameworks).

---

### Q6. How does the TPU interconnect (ICI) enable distributed training?

**Answer:**

TPU Inter-Core Interconnect (ICI) is a custom high-bandwidth, low-latency network that directly connects TPU chips without using external switches. Unlike GPU clusters that rely on NVLink within a node and InfiniBand between nodes, TPU pods use ICI as a unified fabric across all chips.

The ICI topology is a multi-dimensional torus. In TPU v4, this is a 3D torus where each chip has 6 links (2 per dimension: one to the left neighbor, one to the right neighbor). A 4x4x4 pod has 64 chips, and larger pods are constructed by connecting multiple 4x4x4 cubes.

The torus topology has several advantages for collective communication:
1. **Uniform bandwidth**: Every pair of chips has a communication path through the torus with bounded hop count.
2. **Efficient all-reduce**: Ring all-reduce maps naturally onto torus dimensions, and the 3D torus allows decomposing the all-reduce into three 1D ring all-reduces, one per dimension.
3. **No switch bottleneck**: Direct chip-to-chip connections avoid the bandwidth and latency overhead of switch chips.

Google's XLA compiler automatically maps collective operations (all-reduce, all-gather, reduce-scatter) onto the torus topology, choosing the optimal decomposition based on the tensor dimensions and the parallelism strategy (data parallel, model parallel, pipeline parallel).

A key architectural decision is that ICI bandwidth is provisioned to balance with compute throughput. If each chip produces gradients at rate R (FLOPS of backward pass / time), the ICI must provide sufficient bandwidth to all-reduce those gradients in approximately the same time, achieving compute-communication overlap. Google has stated that TPU pods achieve near-linear scaling efficiency (90%+ for data-parallel training) up to thousands of chips, validating the ICI bandwidth provisioning.

---

### Q7. What are the advantages and disadvantages of a systolic array vs a GPU tensor core approach?

**Answer:**

**Systolic array advantages:**
- **Energy efficiency**: Data flows between nearest-neighbor PEs, minimizing data movement distance. Each operand is read from memory once and reused across many PEs as it flows through the array. Energy per operation can be 5-10x lower than a GPU tensor core.
- **Simplicity**: The fixed dataflow eliminates the need for complex instruction scheduling, register files, and memory hierarchy management within the array. This simplicity translates to smaller area per MAC unit and higher compute density.
- **Deterministic timing**: Data movement follows a fixed schedule, making it easy to predict execution time and pipeline the loading of the next matrix tile.

**Systolic array disadvantages:**
- **Utilization sensitivity**: A 256x256 systolic array achieves high utilization only when matrix dimensions are multiples of 256. Smaller or irregular matrix sizes waste PEs. GPUs can assign different-sized problems to different SMs.
- **Fill/drain overhead**: The startup (fill) and completion (drain) phases of the systolic pipeline waste cycles. This is significant for small matrices but amortized for large ones.
- **Limited flexibility**: The fixed dataflow pattern is optimized for matrix multiplication. Non-GEMM operations (softmax, normalization, non-standard attention patterns) must be handled by separate hardware.
- **Programming model**: Systolic arrays require careful scheduling to keep the pipeline full. Irregular computation patterns (dynamic shapes, conditional execution) are difficult to map efficiently.

**GPU tensor core advantages:**
- **Flexibility**: Tensor cores coexist with CUDA cores, shared memory, and a rich memory hierarchy. Non-GEMM operations execute on the same chip using CUDA cores.
- **Programmability**: CUDA provides a mature, flexible programming model. Custom kernels can combine tensor core operations with arbitrary CUDA code.
- **Dynamic workloads**: The warp scheduler dynamically assigns work to execution units, adapting to variable workload patterns.

**GPU tensor core disadvantages:**
- **Energy overhead**: The GPU's general-purpose infrastructure (register files, warp schedulers, cache hierarchy, interconnect) consumes significant area and power.
- **Memory hierarchy complexity**: Achieving peak tensor core utilization requires careful management of shared memory, register usage, and tiling, making optimization challenging.

In practice, both approaches achieve high performance on transformer workloads. The choice often depends on ecosystem factors (CUDA dominance) as much as raw hardware efficiency.

---

### Q8. How does the TPU's software stack (XLA/JAX) differ from CUDA?

**Answer:**

The TPU software stack is built around XLA (Accelerated Linear Algebra) as the compiler and JAX (and TensorFlow) as the programming frameworks. This stack differs from CUDA in philosophy and design:

**XLA compiler**: XLA takes a high-level computation graph (expressed in HLO, High-Level Operations) and compiles it into optimized machine code for the TPU. It performs whole-program optimization including: operator fusion (combining multiple operations into single kernels), memory layout optimization (choosing row-major vs column-major based on downstream consumers), buffer allocation (minimizing peak memory usage), and communication scheduling (overlapping ICI transfers with computation). The programmer writes high-level code; XLA makes the hardware-specific decisions.

**CUDA approach**: NVIDIA's stack gives the programmer more control. Libraries like cuBLAS and cuDNN provide optimized kernels for specific operations, but the programmer (or framework) decides which kernels to call, how to manage memory, and how to schedule communication. Custom CUDA kernels allow fine-grained optimization but require hardware expertise.

| Aspect | TPU (XLA/JAX) | GPU (CUDA) |
|---|---|---|
| Optimization level | Whole-program, compiler-driven | Kernel-level, library + programmer |
| Programmability | High-level (Python/JAX) | Low-level control available (CUDA C++) |
| Custom kernels | Limited (Pallas for TPU kernels) | Full flexibility (CUDA C++, PTX) |
| Operator fusion | Automatic by XLA | Manual or framework-driven (torch.compile) |
| Memory management | Automatic by XLA | Manual (cudaMalloc) or framework-managed |
| Multi-device | Automatic SPMD partitioning (jax.pmap, jax.jit) | Manual (NCCL, torch.distributed) |
| Ecosystem size | Smaller (JAX community) | Massive (CUDA ecosystem) |

The XLA approach trades programmer control for ease of use and portability. It works well for standard workloads (transformers, CNNs) where XLA's automatic optimizations are effective. It is less suited for highly custom workloads that benefit from fine-grained hardware control. Google has addressed this gap with Pallas, a kernel language for writing custom TPU kernels within JAX.

---

### Q9. What matrix dimensions achieve high utilization on a 128x128 systolic array?

**Answer:**

A 128x128 systolic array performing C = A * B where A is M x K and B is K x N achieves maximum utilization when M and N are both multiples of 128 and K is large enough to amortize the fill/drain overhead.

**M and N alignment**: The array computes a 128x128 output tile per pass. If M or N is not a multiple of 128, the excess rows or columns waste PEs. The utilization due to dimension alignment is:

```
Utilization_MN = (ceil(M/128) * 128 * ceil(N/128) * 128) / (ceil(M/128) * 128 * ceil(N/128) * 128)
```

Wait, more precisely, the wasted fraction is:
```
Utilization_MN = (M * N) / (ceil(M/128) * 128 * ceil(N/128) * 128)
```

Examples:
- M=N=128: 100% utilization
- M=N=256: 100% (two tiles per dimension)
- M=N=129: (129*129) / (256*256) = 25.4% utilization (terrible)
- M=N=200: (200*200) / (256*256) = 61.0%
- M=192, N=128: (192*128) / (256*128) = 75.0%

**K dimension**: The inner dimension K determines how many cycles the array is in steady state versus filling/draining. For a single 128x128 output tile with inner dimension K:
```
Fill cycles: 127 + 127 = 254 (for both dimensions)
Compute cycles: K
Utilization_K = K / (K + 254)
```

For K=128: 128/382 = 33.5%. For K=1024: 1024/1278 = 80.1%. For K=4096: 4096/4350 = 94.2%.

**Combined utilization** = Utilization_MN * Utilization_K.

This analysis reveals why modern transformer architectures are designed with dimensions that are powers of 2 or multiples of common tile sizes. The model dimension D=4096 (32 * 128), number of heads H=32, and head dimension D_h=128 are not coincidental -- they ensure high hardware utilization on both GPU tensor cores and TPU systolic arrays.

For inference with batch size B=1 and sequence length S, the effective M dimension is S, which may not be a multiple of 128 for every token position, motivating padding or attention masking strategies that maintain utilization.

---

### Q10. How do TPU pods compare to GPU clusters for large-scale training?

**Answer:**

TPU pods and GPU clusters represent two distinct approaches to large-scale distributed training:

**Interconnect topology**: TPU pods use a direct-connect torus (ICI) with no switches, providing uniform bandwidth between all chips. GPU clusters use a hierarchical network: NVLink within a node (8 GPUs), InfiniBand or Ethernet between nodes. The TPU approach provides more uniform bandwidth; the GPU approach has higher bandwidth within a node but lower bandwidth between nodes.

**Scaling unit**: A TPU v4 pod is a 4096-chip unit with all-to-all ICI connectivity. A GPU cluster is assembled from nodes (typically 8 GPUs each) connected by InfiniBand switches. TPU pods have a fixed topology; GPU clusters can be configured more flexibly.

**Software stack**: TPU pods use XLA with automatic SPMD (Single Program Multiple Data) partitioning. The programmer expresses the computation and XLA maps it onto the pod topology. GPU clusters use NCCL for collective communication and frameworks like Megatron-LM or DeepSpeed for parallelism strategies, giving the programmer more control but requiring more expertise.

**Performance comparison**: Direct comparisons are difficult due to different software stacks and benchmarking methodologies. On MLPerf training benchmarks, both TPU and GPU systems achieve state-of-the-art results. Google's TPU v4 pods have demonstrated strong scaling efficiency on large language models, while NVIDIA's H100 clusters dominate in absolute throughput for many benchmark categories.

**Availability**: GPU clusters can be assembled from commercially available hardware and deployed in any data center. TPU pods are available only through Google Cloud (as Cloud TPU instances). This availability difference significantly impacts adoption: researchers at non-Google organizations overwhelmingly use GPUs.

**Cost model**: Google Cloud TPU pricing is competitive with GPU instances for training workloads where the TPU software stack (JAX/XLA) is well-supported. However, the cost of porting code to JAX/XLA can be significant for teams with existing PyTorch codebases.

In practice, Google and its close collaborators use TPU pods extensively for training frontier models (PaLM, Gemini). Most other organizations use GPU clusters (often NVIDIA DGX or HGX systems), reflecting the dominance of the CUDA ecosystem.
