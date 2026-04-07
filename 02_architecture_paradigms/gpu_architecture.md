# GPU Architecture

This section covers the architecture of modern GPUs for AI workloads, including CUDA cores, tensor cores, streaming multiprocessors, warp scheduling, and key NVIDIA architectures from A100 through B200.

---

### Q1. What is a Streaming Multiprocessor (SM) and how is it organized?

**Answer:**

A Streaming Multiprocessor (SM) is the fundamental compute building block of an NVIDIA GPU. Each SM is an independent processor with its own instruction schedulers, register file, execution units, and shared memory/L1 cache. A GPU contains many SMs (e.g., 132 in H100, 144 in B200) that execute work in parallel.

A modern SM (Hopper architecture) is organized into four processing blocks (also called sub-partitions or quadrants). Each processing block contains: a warp scheduler and dispatch unit, a register file partition (16,384 32-bit registers), 16 FP32 CUDA cores, 16 INT32 units, one tensor core (4th generation), a load/store unit, and a special function unit (SFU) for transcendental functions.

The SM also contains shared resources across all four blocks: 256 KB of combined shared memory and L1 data cache (configurable split), a tex/L1 cache, and a constant cache. The shared memory is accessible by all threads within a thread block (cooperative thread array) running on that SM.

The SM executes instructions in a SIMT (Single Instruction, Multiple Threads) fashion. Threads are grouped into warps of 32 threads that execute the same instruction simultaneously on different data. Each warp scheduler can issue one instruction per cycle to a group of execution units. With four warp schedulers per SM, up to four warps can be in different stages of execution simultaneously, hiding latency through rapid context switching between warps.

---

### Q2. What are tensor cores and how do they differ from CUDA cores?

**Answer:**

CUDA cores are scalar floating-point or integer execution units that perform one FMA (fused multiply-add) operation per clock cycle. They are general-purpose and can execute any arithmetic instruction. A single CUDA core performs 2 FLOPS per cycle (one multiply, one add in the FMA).

Tensor cores are specialized matrix multiply-accumulate units that perform small matrix multiplications in a single operation. A 4th-generation tensor core (Hopper) can compute a 16x8x16 FP16 matrix multiply-accumulate per clock cycle, producing 16x8 = 128 output elements, each requiring 16 multiply-add operations, for a total of 128 * 16 * 2 = 4096 FLOPS per cycle. This represents a roughly 64x throughput advantage over a CUDA core for matrix operations.

Key differences:

| Feature | CUDA Core | Tensor Core |
|---|---|---|
| Operation | Scalar FMA | Matrix MMA |
| Throughput | 2 FLOPS/cycle | ~4096 FLOPS/cycle (FP16) |
| Flexibility | Any arithmetic | Matrix multiply only |
| Supported types | FP64, FP32, FP16, INT32 | FP64, TF32, FP16, BF16, FP8, INT8, INT4 |
| Programming | Individual instructions | Warp-level cooperative (wmma, mma PTX) |
| Data layout | Any | Specific fragment layouts |

Tensor cores achieve their throughput advantage by exploiting the regularity of matrix multiplication. The datapath is hardwired for MAC operations with no branch prediction, instruction decode overhead, or general-purpose flexibility. The tradeoff is that tensor cores can only be used for operations that map to matrix multiplication, which fortunately includes the dominant operations in deep learning.

Programming tensor cores requires using either NVIDIA's libraries (cuBLAS, cuDNN) or warp-level matrix operations (WMMA or MMA PTX instructions). The programmer or library must ensure that data is in the correct fragment layout and that matrix dimensions are multiples of the tensor core's tile size.

---

### Q3. How does warp scheduling work and why is it important for GPU performance?

**Answer:**

Warp scheduling is the mechanism by which a GPU SM selects which warp to execute each cycle. It is the primary means of hiding latency in the GPU architecture.

Each SM can have many warps resident simultaneously (up to 64 on Hopper, with 32 threads per warp = 2048 threads per SM). At any given moment, most warps are stalled waiting for data from memory, waiting for a pipeline stage to complete, or waiting at a synchronization barrier. The warp scheduler selects a warp that has an instruction ready to execute (all operands available, execution unit free) and issues that instruction.

This context-switching between warps happens at zero cost because each warp's registers remain allocated in the register file for the entire duration of the warp's execution. There is no register save/restore overhead, unlike CPU context switching. The GPU trades register file capacity (which is expensive in area) for the ability to hide latency through parallelism.

The concept of occupancy describes the ratio of active warps to the maximum possible warps on an SM. Higher occupancy generally provides more opportunities to hide latency, but it also means each thread gets fewer registers (since the register file is shared among all active threads). A kernel that requires many registers per thread will have lower occupancy.

In practice, maximum occupancy is not always optimal. A kernel with fewer warps but more registers per thread may achieve better performance because each thread can keep more data in registers rather than spilling to slower local memory. The optimal balance depends on the kernel's arithmetic intensity and memory access patterns. Tools like NVIDIA's Occupancy Calculator help determine the optimal launch configuration.

---

### Q4. Describe the evolution from NVIDIA A100 to H100 to B200 for AI workloads.

**Answer:**

**A100 (Ampere, 2020)**: Built on TSMC 7nm, the A100 was the first GPU designed primarily for AI. Key specifications: 6912 FP32 CUDA cores, 432 3rd-gen tensor cores, 80 GB HBM2e at 2.0 TB/s, 312 TFLOPS FP16 tensor, 156 TFLOPS TF32, 624 TOPS INT8. The A100 introduced structured sparsity support (2:4 pattern) that doubled effective tensor core throughput for compatible workloads. TDP was 400W. Third-gen NVLink provided 600 GB/s bidirectional bandwidth.

**H100 (Hopper, 2022)**: Built on TSMC 4nm, the H100 represented a major leap. Key specs: 16,896 FP32 CUDA cores, 528 4th-gen tensor cores, 80 GB HBM3 at 3.35 TB/s, 990 TFLOPS FP16 tensor, 495 TFLOPS TF32, 1979 TOPS INT8. New features included the Transformer Engine with FP8 support (1979 TFLOPS FP8), hardware-accelerated dynamic loss scaling for mixed-precision training, and a DPX instruction set for dynamic programming algorithms. TDP increased to 700W. Fourth-gen NVLink provided 900 GB/s. The H100 also introduced NVLink Switch for connecting up to 256 GPUs in a full-bandwidth NVLink domain.

**B200 (Blackwell, 2024)**: Built on TSMC 4NP (enhanced 4nm), the B200 used a dual-die design connected by a high-bandwidth on-package interconnect, effectively creating a single logical GPU from two dies. Key specs: approximately 2250 TFLOPS FP16, 4500 TFLOPS FP8, 192 GB HBM3e at 8 TB/s, 5th-gen tensor cores with second-generation Transformer Engine. TDP was approximately 1000W (for the full module). Fifth-gen NVLink provided 1800 GB/s. The GB200 superchip combined two B200 GPUs with one Grace CPU connected via NVLink-C2C at 900 GB/s per link.

The generational progression shows: roughly 3x FP16 tensor FLOPS per generation, 1.5-2.5x memory bandwidth per generation, introduction of new lower-precision formats (TF32 in A100, FP8 in H100, microscaling formats in B200), and steadily increasing TDP driving the shift to liquid cooling.

---

### Q5. What is the Transformer Engine and why was it introduced in Hopper?

**Answer:**

The Transformer Engine is a hardware-software feature introduced in the H100 (Hopper) architecture that enables efficient FP8 (8-bit floating-point) computation for transformer workloads while maintaining the accuracy of FP16 or BF16 training.

The challenge with FP8 is that its limited dynamic range (especially the E4M3 format with 4 exponent bits and 3 mantissa bits) makes it prone to overflow and underflow during training. Different layers, different channels, and even different training iterations may require different scaling factors to keep values within the representable range.

The Transformer Engine addresses this with per-tensor dynamic scaling. The hardware tracks the maximum absolute value of activations passing through each tensor core operation and automatically adjusts the scaling factor for subsequent operations. This happens at hardware speed without software intervention, avoiding the overhead of manually inserting scaling operations into the computation graph.

The tensor cores in Hopper support two FP8 formats: E4M3 (4 exponent, 3 mantissa bits, range similar to FP16) used for forward pass activations and weights, and E5M2 (5 exponent, 2 mantissa bits, range similar to BF16) used for gradients in the backward pass. The accumulator is FP32 to prevent accuracy loss from accumulation errors.

The result is that FP8 training can achieve nearly identical model quality to BF16 training while roughly doubling throughput (since FP8 tensor core throughput is 2x FP16 throughput) and halving memory usage for weights and activations. This makes the Transformer Engine one of the most impactful features for large language model training on Hopper and subsequent architectures.

---

### Q6. How does the GPU memory hierarchy work from registers to HBM?

**Answer:**

The GPU memory hierarchy has several levels, each trading capacity for bandwidth and latency:

**Registers**: Each SM has a 256 KB register file (on Hopper). This is the fastest memory (0-cycle access latency, tens of TB/s effective bandwidth) but is private to each thread. With 2048 threads per SM at maximum occupancy, each thread gets 128 32-bit registers. Register pressure is a key performance factor.

**Shared memory / L1 cache**: Each SM has 256 KB of configurable shared memory + L1 cache (on Hopper). Shared memory is explicitly managed by the programmer and shared among all threads in a thread block. It provides approximately 19 TB/s aggregate bandwidth across all SMs. Access latency is approximately 20-30 cycles. Shared memory is organized in 32 banks, and bank conflicts (when multiple threads access different addresses in the same bank) cause serialization.

**L2 cache**: A unified L2 cache shared by all SMs. The H100 has 50 MB of L2 cache with approximately 12 TB/s bandwidth. Access latency is approximately 200 cycles. The L2 acts as a filter for HBM traffic and is particularly important for reducing bandwidth pressure on memory-bound kernels.

**HBM (global memory)**: The main off-chip memory. The H100 has 80 GB of HBM3 at 3.35 TB/s. Access latency is approximately 400-600 cycles. HBM is organized in stacks (5 stacks on H100, each with 8 channels), and interleaving across channels provides the aggregate bandwidth.

For AI workloads, the critical optimization is maximizing data reuse in the upper levels of the hierarchy. GEMM tiling strategies load tiles of the input matrices into shared memory, reuse them across many multiply-accumulate operations, and only access HBM when new tiles are needed. The ratio of compute to HBM accesses (arithmetic intensity) determines whether the kernel is compute-bound or memory-bound, as described in the [roofline model](../01_foundations/compute_fundamentals.md#q3-what-is-the-roofline-model-and-how-is-it-used).

---

### Q7. What is CUDA and how does the programming model map to hardware?

**Answer:**

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel programming model and API for GPU computing. The programming model defines a hierarchy of parallelism that maps to the hardware hierarchy:

**Thread**: The smallest unit of execution. Each thread executes the same kernel function on different data (SIMT model). Threads have private registers and local memory.

**Warp**: 32 threads that execute in lockstep on a single SM processing block. The warp is the fundamental scheduling unit. All threads in a warp execute the same instruction (divergent branches cause serialization).

**Thread block (CTA)**: A group of threads (up to 1024 on modern GPUs) that execute on a single SM and can communicate via shared memory and synchronize via barriers (__syncthreads). Thread blocks are the unit of work assignment to SMs.

**Grid**: A collection of thread blocks that together execute a kernel launch. Thread blocks in a grid are distributed across SMs by the hardware scheduler and cannot communicate except through global memory and atomic operations.

The mapping to hardware:
- Grid -> entire GPU (all SMs)
- Thread block -> one SM (all threads in a block execute on the same SM)
- Warp -> one SM processing block (warp scheduler)
- Thread -> one CUDA core (logically, though physically warps execute in SIMT fashion)

For tensor core programming, CUDA provides warp-level matrix operations (WMMA API or MMA PTX instructions) where a warp of 32 threads cooperatively loads matrix fragments, computes a matrix multiply-accumulate, and stores the result. Higher-level libraries like cuBLAS and cuDNN handle the tiling, data layout, and tensor core dispatch automatically.

The CUDA programming model's success lies in its ability to expose enough parallelism for the hardware to exploit while providing abstractions (shared memory, synchronization, thread blocks) that allow programmers to reason about locality and cooperation.

---

### Q8. What is the role of NVLink and NVSwitch in multi-GPU scaling?

**Answer:**

NVLink is NVIDIA's proprietary high-bandwidth, low-latency interconnect for GPU-to-GPU communication. It provides dramatically higher bandwidth than PCIe, enabling efficient multi-GPU training and inference.

NVLink evolution:
- NVLink 1.0 (Pascal, 2016): 160 GB/s bidirectional per GPU
- NVLink 2.0 (Volta, 2017): 300 GB/s
- NVLink 3.0 (Ampere, 2020): 600 GB/s
- NVLink 4.0 (Hopper, 2022): 900 GB/s
- NVLink 5.0 (Blackwell, 2024): 1800 GB/s

NVSwitch is a dedicated switch chip that enables all-to-all NVLink connectivity among GPUs. Without NVSwitch, NVLink connections are point-to-point, limiting the number of GPUs that can communicate at full bandwidth. NVSwitch enables topologies where every GPU can communicate with every other GPU at full NVLink bandwidth.

In the DGX H100 system, 8 H100 GPUs are connected via 4 NVSwitch chips, providing 900 GB/s all-to-all bandwidth. The NVLink Network (introduced with Hopper) extends this concept with NVLink Switch systems that connect up to 256 GPUs in a single NVLink domain at full bandwidth, enabling efficient all-reduce operations across large clusters without traversing slower InfiniBand or Ethernet networks.

The importance of NVLink for AI workloads stems from the compute-to-communication ratio problem in distributed training. For data-parallel training, gradients must be synchronized via all-reduce. For model-parallel training (tensor parallelism), activations must be communicated between GPUs every layer. In both cases, the communication bandwidth directly impacts training efficiency. NVLink's 10-20x bandwidth advantage over PCIe is often the difference between linear scaling and severe communication bottlenecks.

---

### Q9. How does the A100's structured sparsity support work?

**Answer:**

The A100 introduced hardware support for 2:4 structured sparsity (also called fine-grained structured sparsity), which provides a 2x speedup for tensor core operations on compatible sparse matrices.

The 2:4 pattern requires that in every group of 4 consecutive elements (along the inner dimension of the matrix), exactly 2 are zero. This regularity allows the hardware to compress the matrix to half its dense size using a compact index format: for every group of 4, only the 2 nonzero values and a 2-bit index per nonzero (indicating its position within the group of 4) are stored.

The tensor cores have dedicated hardware to decompress this format on the fly during matrix multiplication. The sparse matrix's nonzero values are loaded from memory (50% fewer bytes), the indices direct the selection of the corresponding elements from the dense matrix, and the multiplication proceeds at full throughput. The result is a 2x improvement in effective throughput and a 2x reduction in memory bandwidth for the sparse operand.

The software workflow involves: (1) training a model normally in dense format, (2) pruning the model to enforce the 2:4 sparsity pattern using magnitude-based or learned pruning, (3) fine-tuning the pruned model to recover accuracy, and (4) running inference using the sparse tensor core path.

In practice, many models can be pruned to 2:4 sparsity with less than 1% accuracy loss after fine-tuning, especially for inference. The 2:4 constraint is more restrictive than general unstructured sparsity (which allows arbitrary zero patterns) but more regular than block sparsity, hitting a sweet spot that enables simple, efficient hardware support.

The H100 and B200 continue to support 2:4 structured sparsity, and the feature has been extended to support FP8 and other data types in newer architectures.

---

### Q10. What are the key differences between consumer GPUs and data center GPUs for AI?

**Answer:**

While consumer GPUs (GeForce/RTX series) and data center GPUs (A100, H100, B200) share the same underlying architecture family, they differ in several critical ways:

**Memory capacity and type**: Data center GPUs use HBM (High Bandwidth Memory) providing 40-192 GB at 2-8 TB/s. Consumer GPUs use GDDR6/GDDR6X providing 8-24 GB at 0.5-1.0 TB/s. The memory capacity difference is the most significant limitation for consumer GPUs in AI: large models simply do not fit.

**ECC memory**: Data center GPUs include ECC (Error Correcting Code) on HBM to ensure correctness during long training runs. A single bit flip during a multi-week training run could corrupt the model. Consumer GPUs typically lack ECC.

**Tensor core capabilities**: Data center GPUs support a broader range of tensor core data types. The H100 supports FP64 tensor cores (important for scientific computing), FP8 with the Transformer Engine, and has higher tensor core counts. Consumer GPUs (e.g., RTX 4090) have tensor cores but with fewer supported formats and lower counts.

**NVLink support**: Data center GPUs support NVLink for high-bandwidth multi-GPU communication. Consumer GPUs are limited to PCIe (and formerly SLI/NVLink on some models), which is 5-10x slower.

**TDP and cooling**: Data center GPUs are designed for sustained high-power operation (400-1000W) with enterprise cooling solutions. Consumer GPUs target 200-450W with consumer cooling.

**Multi-instance GPU (MIG)**: The A100 and H100 support MIG, which partitions a single GPU into up to 7 isolated instances for multi-tenant deployment. Consumer GPUs do not support MIG.

**Reliability and support**: Data center GPUs undergo more rigorous testing, come with longer warranties, and include enterprise support. They are validated for 24/7 operation.

**Price**: Data center GPUs cost $10,000-$40,000+ versus $500-$2,000 for consumer GPUs. The price difference reflects the memory, interconnect, reliability, and support premium.

Despite these differences, consumer GPUs (especially the RTX 4090 with 24 GB and its tensor cores) are widely used for AI research, fine-tuning, and inference of smaller models where the memory capacity limitation is not a constraint.
