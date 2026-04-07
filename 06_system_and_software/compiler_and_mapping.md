# Compiler and Mapping

This section covers the compiler stack for AI accelerators, including graph compilers (XLA, TVM, Triton), operator fusion, tiling strategies, and scheduling.

---

### Q1. What is the role of a graph compiler in the AI accelerator stack?

**Answer:**

A graph compiler takes a high-level computation graph (the neural network expressed as a DAG of operations) and transforms it into optimized machine code for a specific accelerator. The compiler bridges the gap between the framework-level representation (PyTorch, TensorFlow, JAX) and the hardware-specific instruction sequences.

The compiler performs several key optimizations: operation fusion (combining multiple operations into single kernels), memory allocation (minimizing peak memory usage by scheduling buffer lifetimes), tiling and loop optimization (partitioning large operations into hardware-appropriate tiles), layout optimization (choosing data layouts that match hardware access patterns), and scheduling (ordering operations to maximize parallelism and overlap computation with communication).

The importance of the compiler has grown as AI accelerators have become more specialized. A general-purpose GPU with CUDA can fall back on manually optimized library kernels (cuBLAS, cuDNN) for common operations, but novel architectures (TPU, Cerebras, Groq, SambaNova) require compiler-generated code for every operation. The compiler's quality directly determines the accelerator's achieved performance: a poor compiler leaves hardware utilization on the table.

Major graph compilers include XLA (Google, used for TPU and GPU), TVM (Apache, multi-target), Triton (OpenAI, GPU-focused kernel compiler), MLIR (Google, compiler infrastructure), TensorRT (NVIDIA, inference optimization), and IREE (Google, MLIR-based deployment compiler).

---

### Q2. How does operator fusion work and why is it critical for performance?

**Answer:**

Operator fusion combines multiple sequential operations into a single kernel that executes without writing intermediate results to off-chip memory. This is critical because many neural network operations are memory-bound: their execution time is dominated by reading inputs and writing outputs to HBM rather than by computation.

The compiler identifies fusible patterns by analyzing the computation graph. Common fusion patterns include:

**Element-wise chains**: ReLU(MatMul(x, W) + bias) fuses the bias addition and activation function with the matrix multiply's output write. Instead of 3 separate kernels (MatMul -> store -> load -> BiasAdd -> store -> load -> ReLU -> store), the fused kernel computes all three in registers/shared memory with a single store.

**Reduction-element-wise**: LayerNorm (which involves mean computation, variance computation, and normalization) is fused into a single pass over the data, computing all three steps while the data is in registers.

**Attention fusion**: FlashAttention fuses Q*K^T, softmax, and score*V into a single kernel that tiles across the sequence dimension and never materializes the full S*S attention matrix in HBM.

**Epilogue fusion**: After a GEMM, bias addition, activation, and residual connection are fused as an "epilogue" that operates on the GEMM output while it is still in registers or shared memory.

The performance impact of fusion is substantial. For a transformer layer, the non-GEMM operations (activation functions, normalization, residual connections) account for only 5-10% of FLOPS but can consume 20-40% of wall-clock time without fusion due to their memory-bound nature. Fusion can reduce this to near-zero overhead.

---

### Q3. What is Triton and how does it differ from CUDA?

**Answer:**

Triton is an open-source programming language and compiler (developed by OpenAI) for writing efficient GPU kernels at a higher level of abstraction than CUDA. It targets the gap between handwritten CUDA (expert-level performance but difficult) and compiler-generated code (easier but often suboptimal).

Key differences from CUDA:

**Programming model**: Triton programs operate on blocks of data (tiles) rather than individual threads. The programmer specifies tile sizes and operations on tiles; the compiler handles thread mapping, shared memory management, vectorized loads, and register allocation. In CUDA, the programmer must manage all of these explicitly.

**Abstraction level**: A Triton kernel for matrix multiplication is roughly 30 lines of Python-like code, compared to 200+ lines for an optimized CUDA kernel. The compiler automatically generates shared memory tiling, double buffering, vectorized loads, and tensor core usage.

**Auto-tuning**: Triton includes an auto-tuning framework that searches over tile sizes, block sizes, and other parameters to find the best configuration for a given kernel on a given GPU. CUDA optimization typically requires manual tuning by experts.

**Performance**: For many common patterns (GEMM, attention, element-wise fused kernels), Triton achieves 85-100% of handwritten CUDA performance. For unusual or highly irregular patterns, hand-tuned CUDA can still be faster because the programmer has full control over every hardware feature.

**Portability**: Triton targets NVIDIA GPUs through PTX/CUBIN but has experimental backends for AMD GPUs. The tile-based abstraction is more portable than CUDA's thread-level model.

Triton has become the dominant tool for writing custom fused kernels in the PyTorch ecosystem. FlashAttention, many GPTQ/AWQ quantization kernels, and various custom attention patterns are implemented in Triton.

---

### Q4. What is XLA and how does it optimize for TPUs?

**Answer:**

XLA (Accelerated Linear Algebra) is Google's domain-specific compiler for linear algebra operations. It is the primary compiler for Google TPUs and also supports GPUs and CPUs. XLA is used through JAX (directly) and TensorFlow (via tf.function with XLA compilation).

XLA's compilation pipeline:
1. **HLO (High-Level Operations)**: The input representation. Each HLO operation corresponds to a mathematical operation (dot product, convolution, reduce, transpose). The compiler receives the entire computation graph as a sequence of HLO instructions.
2. **HLO optimization passes**: Algebraic simplification, constant folding, dead code elimination, common subexpression elimination, operation fusion, and layout assignment.
3. **Target-specific lowering**: Converts optimized HLO into target-specific instructions. For TPU: MXU operations, vector unit operations, DMA transfers, and ICI communication. For GPU: CUDA kernels or calls to cuBLAS/cuDNN.
4. **Buffer assignment**: Allocates on-chip memory for all intermediate tensors, minimizing peak memory usage through liveness analysis and buffer sharing.
5. **Scheduling**: Orders operations to overlap computation with memory transfers and ICI communication.

TPU-specific optimizations in XLA include: automatic SPMD (Single Program Multiple Data) partitioning across TPU cores and pods, tiling for the MXU systolic array dimensions (128x128), fusion of operations into efficient MXU-VPU (Vector Processing Unit) sequences, and communication scheduling for ICI collective operations.

XLA's whole-program compilation approach (compiling the entire computation graph at once) enables global optimizations that per-kernel compilers cannot: cross-operation tiling (choosing compatible tile sizes across a sequence of operations), global memory allocation (sharing buffers between operations that do not overlap in lifetime), and end-to-end communication scheduling.

---

### Q5. What is TVM and what is its approach to code generation?

**Answer:**

TVM (Tensor Virtual Machine) is an open-source compiler framework for deep learning workloads that targets a wide range of hardware backends including GPUs, CPUs, FPGAs, and custom accelerators. Its distinctive approach is the separation of computation definition from schedule optimization.

**Compute definition**: The programmer defines what is computed using a mathematical tensor expression language. For example, a matrix multiply is expressed as: `C[i, j] = sum(A[i, k] * B[k, j], axis=k)`. This is a pure mathematical specification with no implementation details.

**Schedule**: The schedule specifies how the computation is executed: loop tiling, loop ordering, parallelization, vectorization, unrolling, and memory hierarchy mapping. TVM provides a rich set of schedule primitives that transform the loop nest. Different schedules of the same computation produce different code with vastly different performance.

**Auto-scheduling (AutoTVM, Ansor)**: TVM includes automated schedule search engines that explore the space of possible schedules and select the best one based on measured performance on the target hardware. AutoTVM uses a machine learning model to guide the search; Ansor (also called Auto-Scheduler) generates schedules from sketch templates.

**Relay IR**: TVM's high-level intermediate representation for computation graphs. Relay performs graph-level optimizations (operator fusion, constant folding, layout transformation) before lowering to the tensor expression level for schedule optimization.

**Target support**: TVM generates code for CUDA, OpenCL, LLVM (for CPUs), Verilog (for FPGAs via VTA), and custom accelerator backends via the BYOC (Bring Your Own Codegen) framework. This makes TVM valuable for novel accelerator designs that lack vendor-provided libraries.

TVM's strength is its ability to generate competitive code for diverse hardware without relying on vendor-specific libraries. Its weakness is that the automated schedule search can be time-consuming (hours for complex operators) and may not always find schedules as good as hand-tuned implementations.

---

### Q6. How does the compiler determine optimal tiling for a given hardware target?

**Answer:**

Tiling optimization involves choosing tile sizes (Tm, Tn, Tk for a GEMM, or spatial tile sizes for convolution) that maximize performance on the target hardware. The compiler considers several constraints:

**On-chip memory capacity**: Tiles of input operands must fit in shared memory/scratchpad, with room for double buffering. The constraint is: 2*(Tm*Tk + Tk*Tn) + Tm*Tn*accumulator_bytes <= SRAM_capacity.

**Compute unit dimensions**: Tensor cores or systolic arrays have fixed tile sizes (e.g., 16x8x16 for Hopper tensor cores). The outer tiles should be multiples of these hardware tile sizes to avoid waste.

**Parallelism**: The number of independent tiles should be large enough to keep all compute units busy. For 132 SMs, there should be at least 132 output tiles (and preferably 2-4x more for scheduling flexibility).

**Arithmetic intensity**: Larger tiles increase arithmetic intensity (more compute per byte loaded from HBM). The tile size should push the workload's arithmetic intensity above the ridge point.

**Register pressure**: The innermost loop's operands and accumulators must fit in registers. Too large an inner tile spills to shared memory, adding overhead.

The optimization is typically formulated as a constrained optimization problem:
```
Maximize: arithmetic_intensity(Tm, Tn, Tk)
Subject to: SRAM_constraint(Tm, Tn, Tk) <= SRAM_capacity
            Tm % hardware_tile_m == 0
            Tn % hardware_tile_n == 0
            Tk % hardware_tile_k == 0
            parallelism(Tm, Tn) >= num_compute_units
```

In practice, compilers solve this through: analytical models (closed-form solutions for special cases), exhaustive search over a discrete set of candidate tile sizes, or auto-tuning (measuring actual performance for each candidate on the target hardware). Modern compilers like Triton and TVM use auto-tuning extensively.

---

### Q7. What is MLIR and why is it important for AI compiler development?

**Answer:**

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure project developed at Google that provides a framework for building domain-specific compilers with multiple levels of abstraction. It is important for AI compilers because it addresses the "N*M problem": N different frameworks (PyTorch, TensorFlow, JAX) targeting M different hardware backends (GPU, TPU, custom ASICs) would traditionally require N*M separate compilers.

MLIR provides a common infrastructure where each level of abstraction is represented as a "dialect" (a set of operations at a particular abstraction level). The compilation process lowers operations through progressively more concrete dialects:

1. **High-level dialect**: Operation names like "conv2d", "matmul", "softmax" (similar to XLA HLO or ONNX).
2. **Linalg dialect**: Named linear algebra operations with explicit loop structure.
3. **Affine/SCF dialect**: Explicit loop nests with affine loop bounds and control flow.
4. **Vector dialect**: Vector operations targeting SIMD units.
5. **LLVM/GPU dialect**: Low-level instructions targeting specific hardware.

Each framework provides a lowering from its graph representation to MLIR's high-level dialect. Each hardware backend provides a lowering from MLIR's low-level dialect to machine code. The intermediate transformations (tiling, fusion, vectorization) are shared across all framework-backend combinations.

MLIR has been adopted by: TensorFlow (via MLIR-HLO), JAX/XLA (via StableHLO), PyTorch (via torch-mlir), IREE (Google's deployment compiler), and various hardware startups building custom accelerator compilers. Its modular design enables rapid development of new compiler backends for novel hardware.

---

### Q8. How does the compiler handle memory allocation and scheduling?

**Answer:**

Memory allocation and scheduling are tightly coupled optimizations that determine when operations execute and where their data resides:

**Liveness analysis**: The compiler determines the lifetime of each tensor (from when it is first produced to when it is last consumed). Tensors with non-overlapping lifetimes can share the same memory buffer, reducing peak memory usage. For a computation graph with N tensors, the minimum memory is the maximum sum of live tensor sizes at any point in the execution schedule.

**Buffer sharing/reuse**: The compiler assigns tensors to memory locations, reusing buffers when safe. This is similar to register allocation in traditional compilers but for larger data structures. Graph coloring algorithms are commonly used: tensors that interfere (have overlapping lifetimes) get different buffers; non-interfering tensors can share.

**Schedule optimization**: The order in which operations execute affects both memory usage and performance. Executing operations in a depth-first order (completing one branch of the computation graph before starting another) tends to minimize peak memory (each branch's intermediate tensors are freed before the next branch starts). Breadth-first order may expose more parallelism but increases peak memory.

**Memory hierarchy placement**: The compiler decides which tensors reside in which level of the memory hierarchy. Hot tensors (accessed frequently) are placed in faster memory (shared memory/registers). Cold tensors (accessed once) are placed in slower memory (HBM). For double buffering, the compiler alternates tensor placement between two buffer regions.

**DMA scheduling**: For accelerators with explicit DMA (systolic arrays, dataflow architectures), the compiler schedules DMA transfers to overlap data loading with computation. The DMA schedule must ensure that data arrives before it is needed (no stalls) and that buffers are freed after consumption (no overflow).

---

### Q9. What is torch.compile and how does it relate to the compiler stack?

**Answer:**

torch.compile is PyTorch's JIT (Just-In-Time) compilation system introduced in PyTorch 2.0. It captures a PyTorch model's computation graph at runtime and compiles it into optimized code, bridging the gap between PyTorch's eager execution model and the ahead-of-time compilation of frameworks like JAX/XLA.

The compilation pipeline has three stages:

**TorchDynamo (graph capture)**: A Python bytecode interceptor that captures the computation graph by analyzing the Python bytecode of the model's forward pass. It handles control flow (loops, conditionals) by creating graph breaks where dynamic Python behavior occurs and compiling the static graph segments.

**TorchInductor (code generation)**: The default compiler backend that generates optimized Triton kernels for GPU execution. Inductor performs operator fusion (combining element-wise operations, reductions, and GEMM epilogues), memory planning, and Triton kernel generation with auto-tuning.

**Alternative backends**: torch.compile supports pluggable backends including: TensorRT (NVIDIA inference optimization), ONNX Runtime, XLA (for TPU compatibility), and custom backends for novel hardware.

The significance of torch.compile for the AI accelerator ecosystem is that it provides a standard compilation entry point for PyTorch, which is the dominant research framework. Hardware vendors can write a torch.compile backend to support their accelerator without requiring users to rewrite their PyTorch code. This lowers the barrier to adoption for new hardware.

Performance impact: torch.compile typically provides 1.3-2x speedup over eager PyTorch execution, primarily from operator fusion and reduced Python overhead. The speedup varies by model: models with many small operations benefit more from fusion, while models dominated by large GEMMs (which already call optimized cuBLAS) benefit less.

---

### Q10. How do compilers handle dynamic shapes and control flow?

**Answer:**

Dynamic shapes (tensors whose dimensions are not known until runtime) and data-dependent control flow (if-statements whose condition depends on tensor values) are challenging for AI compilers because most optimizations assume static, compile-time-known shapes and deterministic execution paths.

**Challenges:**
- Tiling decisions depend on tensor dimensions. With dynamic shapes, the optimal tile size changes at runtime.
- Memory allocation requires knowing tensor sizes. Dynamic shapes prevent ahead-of-time buffer allocation.
- Operator fusion may be valid only for certain shape combinations.
- Static scheduling (as in Groq) is impossible with dynamic shapes because the schedule depends on dimensions.

**Solutions:**

**Shape specialization**: The compiler compiles separate optimized versions for different shape combinations and dispatches to the appropriate version at runtime. XLA's "auto-clustering" and torch.compile's "guards" implement this: the compiled code includes shape checks, and if the shapes change, the code is recompiled or falls back to a generic version.

**Symbolic shapes**: The compiler treats dimensions as symbolic variables and generates code that is parameterized by shape. For example, tile counts are computed at runtime as ceil(M/Tm), but the tiled loop structure is fixed. PyTorch's torch.compile supports symbolic shapes through the torch.fx framework.

**Padding**: Dynamic dimensions are padded to the next multiple of a convenient size (e.g., multiples of 128 for tensor cores). This allows using a fixed tile size at the cost of wasted computation on padding elements.

**Bucketing**: Input dimensions are grouped into buckets (e.g., sequence lengths 1-128, 129-256, 257-512), and a separate compiled kernel is generated for each bucket. This limits the number of recompilations while handling variable dimensions.

For most AI inference deployments, shapes are static or drawn from a small set of known configurations, making shape specialization effective. For training with variable-length sequences, bucketing and padding are the standard approaches.
