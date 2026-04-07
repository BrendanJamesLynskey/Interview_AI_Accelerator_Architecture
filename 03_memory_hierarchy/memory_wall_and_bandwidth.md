# Memory Wall and Bandwidth

This section covers the memory bandwidth bottleneck in AI accelerators, data reuse strategies, and tiling techniques used to maximize performance.

---

### Q1. What is the memory wall and why is it the central challenge in AI accelerator design?

**Answer:**

The memory wall refers to the growing disparity between processor compute throughput and memory bandwidth. Compute throughput has historically doubled every 2-3 years, while memory bandwidth has improved at only 1.3-1.5x per generation. This means that an increasing fraction of workloads are limited not by the rate of computation but by the rate at which data can be delivered to the compute units.

For AI accelerators, this manifests acutely. An NVIDIA H100 can perform 990 TFLOPS of FP16 computation but has only 3.35 TB/s of HBM bandwidth. To keep the compute units fully utilized, each byte fetched from HBM must support approximately 295 floating-point operations (the ridge point). Any operation with lower arithmetic intensity leaves compute units idle, wasting the silicon area and power invested in those units.

The memory wall drives nearly every major architectural decision in AI accelerator design: the size of on-chip SRAM (larger SRAM reduces HBM accesses), the choice of data types (FP8 and INT8 double effective data density), the design of the memory hierarchy (multi-level buffering to maximize reuse), operator fusion strategies (keeping intermediate results on-chip), and even model architecture choices (preferring operations with high arithmetic intensity). Understanding the memory wall and its implications is essential for any role in AI hardware design.

---

### Q2. What are the three types of data reuse in neural network computations?

**Answer:**

Data reuse refers to accessing the same data element multiple times to perform different computations, thereby amortizing the cost of fetching that element from memory. In neural network computations, there are three fundamental types:

**Temporal reuse**: The same data element is used at different points in time by the same processing element. For example, in a matrix multiplication C = A * B, a single weight element B[k][j] is multiplied with every element in column k of A across all rows. If the PE stores B[k][j] in a local register, it can reuse it across M rows of A without re-fetching. The amount of temporal reuse equals the number of times the element is used.

**Spatial reuse**: The same data element is used simultaneously by different processing elements. In a systolic array computing a GEMM, a single activation element A[i][k] flows across an entire row of PEs, being multiplied with different weights at each PE. The element is fetched from memory once but used N times (once per PE in the row).

**Convolutional reuse (specific to CNNs)**: In convolution, the same input activation is part of multiple overlapping receptive fields. A single pixel at position (x, y) contributes to the computation of multiple output pixels (those whose filter footprint includes (x, y)). This reuse is proportional to the filter size: a KxK filter provides approximately K^2 reuse of each input element.

Maximizing all three types of reuse simultaneously is the goal of dataflow optimization. The row-stationary dataflow (Eyeriss) was specifically designed to maximize all three types. In practice, the [tiling strategy](../03_memory_hierarchy/memory_wall_and_bandwidth.md#q5-what-is-tiling-and-how-does-it-improve-memory-efficiency) determines how much reuse is captured at each level of the memory hierarchy.

---

### Q3. How does the memory bandwidth requirement differ between training and inference?

**Answer:**

Training and inference have fundamentally different memory bandwidth characteristics:

**Training** involves three phases per layer: forward pass, backward pass (computing activation gradients), and weight gradient computation. Training typically uses large batch sizes to achieve high compute utilization. The large batch size increases the arithmetic intensity of GEMMs (more input rows reuse the same weights), making training workloads predominantly compute-bound on modern hardware. Memory bandwidth is important for loading weights and storing activations for the backward pass, but compute throughput is usually the binding constraint.

**Inference** often operates with batch size 1 or small batches (especially for latency-sensitive applications like chat). At batch size 1, every fully connected layer degenerates to a matrix-vector multiply where the weight matrix must be loaded from memory but is used for only one input vector. The arithmetic intensity is approximately 2 FLOPS per weight element loaded (one multiply, one add), making inference extremely memory-bandwidth-bound.

Quantitative comparison for a single linear layer (D=4096):

| Metric | Training (B=256) | Inference (B=1) |
|---|---|---|
| FLOPS | 2 * 256 * 4096^2 = 8.6T | 2 * 1 * 4096^2 = 33.6M |
| Weight bytes (FP16) | 4096^2 * 2 = 32 MB | 4096^2 * 2 = 32 MB |
| Arithmetic intensity | 268 FLOPS/byte | 1.05 FLOPS/byte |
| Bottleneck | Compute | Memory bandwidth |

This difference explains why inference-focused accelerators (Groq LPU, AWS Inferentia) may prioritize memory bandwidth and capacity over raw compute throughput, and why techniques like weight quantization (INT8, INT4) are so impactful for inference: they directly reduce the memory bandwidth requirement.

---

### Q4. What is the bandwidth hierarchy in a modern AI accelerator?

**Answer:**

A modern AI accelerator has a multi-level memory hierarchy where each level trades capacity for bandwidth:

| Level | Capacity | Bandwidth | Latency | Energy per access |
|---|---|---|---|---|
| Register file | ~256 KB per SM | ~20 TB/s (per SM) | 0 cycles | ~1 pJ |
| Shared memory / L1 | ~256 KB per SM | ~19 TB/s (aggregate) | ~20-30 cycles | ~5 pJ |
| L2 cache | ~50 MB | ~12 TB/s | ~200 cycles | ~50 pJ |
| HBM (off-chip) | 80-192 GB | 2-8 TB/s | ~400-600 cycles | ~200 pJ |
| Host memory (PCIe) | TBs | 64-128 GB/s | ~10,000+ cycles | ~1000+ pJ |
| Network (InfiniBand) | Distributed | 50-100 GB/s | ~1,000,000+ ns | N/A |

(Values approximate for NVIDIA H100.)

The energy per access column is particularly important: accessing a byte from HBM costs roughly 200x more energy than accessing it from a register. Since energy efficiency (FLOPS/W) is the key metric for data center accelerators, minimizing the number of accesses to lower levels of the hierarchy is critical.

The design challenge is that each level's capacity limits how much data can be reused before accessing the next level. A 256 KB shared memory can hold only a small tile of a weight matrix. The tiling strategy must be designed so that the tile size matches the available SRAM capacity while maximizing the compute performed on each tile before it is evicted.

For AI accelerators compared to CPUs, the hierarchy is often flatter (fewer levels) and relies more on explicit management (scratchpads, DMA transfers) rather than implicit caching. This gives the software/compiler more control over data placement at the cost of programming complexity.

---

### Q5. What is tiling and how does it improve memory efficiency?

**Answer:**

Tiling (also called blocking or loop tiling) is a technique that partitions a large computation into smaller tiles that fit in on-chip memory, maximizing data reuse before the data is evicted. It is the single most important optimization for matrix multiplication on hardware with limited on-chip memory.

Consider a matrix multiplication C = A * B where A is M x K, B is K x N. Without tiling, the naive algorithm loads each element from HBM every time it is needed, resulting in O(M*K*N) HBM accesses.

With tiling, the matrices are partitioned into tiles: A into (M/Tm) x (K/Tk) tiles of size Tm x Tk, B into (K/Tk) x (N/Tn) tiles of size Tk x Tn. The computation proceeds:

```python
for m in range(0, M, Tm):
    for n in range(0, N, Tn):
        C_tile = zeros(Tm, Tn)  # in registers or shared memory
        for k in range(0, K, Tk):
            A_tile = load(A[m:m+Tm, k:k+Tk])   # from HBM to SRAM
            B_tile = load(B[k:k+Tk, n:n+Tn])   # from HBM to SRAM
            C_tile += A_tile @ B_tile            # computed in SRAM
        store(C[m:m+Tm, n:n+Tn], C_tile)       # SRAM to HBM
```

The total HBM accesses with tiling:
- A tiles loaded: (M/Tm) * (N/Tn) * (K/Tk) * Tm * Tk = M * K * N / Tn
- B tiles loaded: (M/Tm) * (N/Tn) * (K/Tk) * Tk * Tn = M * K * N / Tm
- Total: M*K*N * (1/Tn + 1/Tm)

The SRAM requirement is Tm * Tk + Tk * Tn + Tm * Tn elements. Given a fixed SRAM budget S, the optimal tile sizes minimize HBM accesses while fitting within S. For square tiles (Tm = Tn = T), HBM accesses scale as 2*M*K*N/T, and the SRAM requirement is approximately 2*T*Tk + T^2. Larger tiles mean fewer HBM accesses but require more SRAM.

This tiling analysis is fundamental to AI accelerator interviews. Being able to derive the optimal tile sizes for a given SRAM budget and matrix dimensions is a commonly asked interview problem.

---

### Q6. What is double buffering and why is it important?

**Answer:**

Double buffering is a technique that overlaps data loading with computation by maintaining two copies (buffers) of the data: one being used for computation and one being loaded with the next set of data. When computation on the first buffer finishes, the roles swap, and computation proceeds on the freshly loaded buffer while the first buffer is loaded with the next data.

Without double buffering, the execution timeline alternates between load and compute phases:
```
[Load tile 0] [Compute tile 0] [Load tile 1] [Compute tile 1] ...
```

With double buffering:
```
[Load tile 0] [Compute tile 0 | Load tile 1] [Compute tile 1 | Load tile 2] ...
```

The load time is hidden behind the compute time (assuming compute time >= load time, which is true for compute-bound operations).

The cost of double buffering is that the on-chip SRAM requirement doubles: two complete tile sets must be stored simultaneously. For a GEMM tile of size Tm x Tk for A and Tk x Tn for B, double buffering requires 2 * (Tm * Tk + Tk * Tn) elements of SRAM.

This means the effective tile size must be reduced to fit two buffers in the same SRAM, which increases the number of tiles and potentially increases total HBM accesses. There is a tradeoff:
- Larger tiles: fewer HBM accesses, but compute stalls waiting for loads (no double buffering)
- Smaller tiles with double buffering: more HBM accesses, but load latency is hidden

In practice, double buffering (or even triple buffering for deeply pipelined systems) is almost universally used because the latency hiding benefit outweighs the reduced tile size. The compiler or programmer must carefully choose tile sizes that balance these concerns for the specific hardware's SRAM capacity and HBM bandwidth.

---

### Q7. How does operator fusion reduce memory bandwidth requirements?

**Answer:**

Operator fusion combines multiple sequential operations into a single kernel that executes without writing intermediate results to off-chip memory. This is critical because many operations in neural networks are memory-bound and their intermediate results are consumed only by the immediately following operation.

Consider a sequence: GEMM -> Bias Add -> ReLU -> Layer Norm. Without fusion:
1. GEMM writes output to HBM (write: B*S*D elements)
2. Bias Add reads from HBM, adds bias, writes to HBM (read + write: 2 * B*S*D)
3. ReLU reads from HBM, applies ReLU, writes to HBM (read + write: 2 * B*S*D)
4. Layer Norm reads from HBM, normalizes, writes to HBM (read + write: 2 * B*S*D)

Total HBM traffic: 7 * B*S*D elements (1 write + 3 * (read + write))

With full fusion:
1. GEMM computes output in shared memory/registers
2. Bias add applied in registers
3. ReLU applied in registers
4. Layer Norm computed (requires reduction, but can be done in shared memory)
5. Final result written to HBM

Total HBM traffic: 1 * B*S*D elements (just the final write) + input reads

Fusion reduces HBM traffic by approximately 7x in this example. Since Bias Add, ReLU, and Layer Norm are all memory-bound (arithmetic intensity near 1 FLOP/byte), eliminating their HBM accesses can significantly reduce total execution time.

FlashAttention is the most impactful example of fusion in modern AI: it fuses the entire attention computation (Q*K^T, softmax, score*V) into a single kernel that tiles across the sequence dimension and never materializes the full S x S attention matrix in HBM. This reduces attention HBM traffic from O(S^2) to O(S), enabling efficient long-context models.

Modern compiler stacks (XLA, torch.compile, Triton) perform automatic fusion as a key optimization pass. The quality of fusion decisions directly impacts end-to-end performance.

---

### Q8. What is the bandwidth cost of the KV cache during autoregressive inference?

**Answer:**

In autoregressive language model inference, each new token generation requires attending to all previous tokens. The key and value tensors from previous tokens are stored in the KV cache to avoid recomputing them. This cache creates a significant memory bandwidth burden.

For a model with L layers, H attention heads, head dimension D_h, and current sequence length S (number of tokens generated so far):

**KV cache size:**
```
Size = 2 * L * H * D_h * S * bytes_per_element
```

For a 70B parameter model (L=80, H=64, D_h=128) at sequence length S=4096 with FP16:
```
Size = 2 * 80 * 64 * 128 * 4096 * 2 bytes = 10.7 GB
```

**Bandwidth per token:** To generate each new token, the entire KV cache must be read from HBM (to compute attention scores against all previous tokens):
```
Bandwidth per token = 2 * L * H * D_h * S * 2 bytes
```

At S=4096: 10.7 GB read per token. On an H100 with 3.35 TB/s bandwidth:
```
Time per token (KV cache read alone) = 10.7 GB / 3.35 TB/s = 3.2 ms
```

This is often the dominant latency component in autoregressive inference. The KV cache bandwidth grows linearly with sequence length, making long-context inference increasingly memory-bound.

Optimization strategies include:
- **KV cache quantization**: Storing cached K/V in INT8 or FP8, reducing bandwidth by 2x with minimal accuracy impact
- **Multi-Query Attention (MQA)**: Using a single K/V head shared across all query heads, reducing KV cache by H/1 = 64x
- **Grouped-Query Attention (GQA)**: A compromise where K/V heads are shared in groups, reducing KV cache by the group factor (e.g., 8x)
- **PagedAttention (vLLM)**: Efficient memory management for variable-length KV caches
- **Sliding window attention**: Limiting attention to recent tokens, capping KV cache size

---

### Q9. What is prefetching and how is it applied in AI accelerators?

**Answer:**

Prefetching is the technique of initiating data transfers from slower memory (HBM) to faster memory (SRAM, registers) before the data is actually needed by the compute units, so that the data arrives just in time to be consumed without stalling computation.

In AI accelerators, prefetching is implemented through several mechanisms:

**Software prefetching (DMA-initiated)**: The compiler or runtime explicitly schedules DMA (Direct Memory Access) transfers to load the next tile of data from HBM to SRAM while the current tile is being processed. This is essentially double buffering with explicit DMA scheduling. The DMA engine operates independently of the compute units, allowing overlap.

**Hardware prefetching**: Some accelerators include hardware prefetch logic that detects strided access patterns and initiates fetches ahead of the compute unit's demand. This is common in CPUs but less common in AI accelerators, which rely more on software-scheduled transfers due to their predictable access patterns.

**Pipeline prefetching**: In a systolic array or dataflow architecture, data is continuously streamed into the array from buffers that are themselves being filled from HBM. The multi-stage pipeline (HBM -> global buffer -> local PE buffer -> PE) acts as a prefetch chain where each stage provides data to the next.

Effective prefetching requires accurate knowledge of future data needs, which is straightforward for AI workloads with their regular, predictable access patterns. The compiler computes the data needed for each computation phase and schedules transfers accordingly.

The challenge is balancing prefetch distance with buffer space. Prefetching too early wastes SRAM capacity (the data occupies buffer space while waiting to be used). Prefetching too late causes stalls. The optimal prefetch distance depends on the HBM latency (typically 400-600 ns) and the compute time per tile.

---

### Q10. How do you estimate the minimum memory bandwidth required for a given workload?

**Answer:**

The minimum bandwidth requirement is determined by the workload's data movement needs and the target execution time:

```
Minimum bandwidth = Total bytes to move / Target execution time
```

For compute-bound operations, the target execution time is determined by compute throughput:
```
Target time = Total FLOPS / Sustained TFLOPS
Minimum bandwidth = Total bytes / Target time
```

For a large GEMM (M=4096, K=4096, N=4096, FP16):
```
Total FLOPS = 2 * 4096^3 = 1.37 * 10^11
Total bytes = (4096^2 + 4096^2 + 4096^2) * 2 = 100.7 MB (A + B + C)
Target time on H100 = 1.37 * 10^11 / (990 * 10^12) = 0.139 ms
Minimum bandwidth = 100.7 MB / 0.139 ms = 724 GB/s
```

This is well within the H100's 3.35 TB/s, confirming the GEMM is compute-bound.

For memory-bound operations, the minimum bandwidth equals the application's required throughput:
```
Minimum bandwidth = bytes_per_inference * inferences_per_second
```

For serving a 70B parameter model at 100 tokens/second with INT8 weights:
```
Weight reads per token = 70 * 10^9 bytes = 70 GB
Minimum bandwidth = 70 GB * 100 = 7 TB/s
```

This exceeds a single H100's bandwidth (3.35 TB/s), indicating that either multiple GPUs with tensor parallelism (splitting weights across GPUs) or weight quantization (INT4 to halve the bandwidth) is needed.

This kind of back-of-envelope bandwidth calculation is essential for early-stage architecture decisions: how much HBM bandwidth to provision, how many chips to use, and what precision to target.
