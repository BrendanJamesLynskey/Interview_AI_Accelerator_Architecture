# Workload Characteristics

This section covers the computational characteristics of the major deep learning workloads, including CNNs, transformers, attention mechanisms, MLPs, GEMMs, and how convolution can be expressed as GEMM.

---

### Q1. What are the dominant compute patterns in deep learning workloads?

**Answer:**

Deep learning workloads are dominated by a small number of fundamental operations that account for the vast majority of compute time:

**General Matrix Multiply (GEMM)**: The single most important operation. Fully connected layers, attention projections (Q, K, V), and feed-forward network layers are all GEMMs. Even convolutions are commonly lowered to GEMMs for efficient execution on hardware with matrix multiply units.

**Convolution**: The defining operation of CNNs, where a small filter kernel slides across a spatial input, computing dot products at each position. Although logically distinct from matrix multiplication, it is almost always implemented as a GEMM (via im2col or similar transformations) or as a specialized direct convolution on hardware with native convolution support.

**Element-wise operations**: Activation functions (ReLU, GELU, SiLU), residual additions, scaling, and masking. These have trivially low arithmetic intensity (one or a few ops per element loaded) and are almost always memory-bound.

**Reduction operations**: Softmax, layer normalization, batch normalization, and loss computation involve reducing across one or more dimensions. These are also typically memory-bound.

**Attention score computation**: The Q*K^T matrix multiply followed by softmax and multiplication by V. This has unique characteristics due to its quadratic scaling with sequence length.

The key architectural insight is that GEMMs dominate compute time but element-wise and reduction operations can dominate wall-clock time if they are not efficiently fused or if memory bandwidth is insufficient. A well-designed accelerator must handle both efficiently.

---

### Q2. How does convolution map to matrix multiplication (im2col)?

**Answer:**

The im2col (image-to-column) transformation converts a convolution operation into a GEMM by rearranging the input data. Consider a 2D convolution with input of shape (C_in, H, W), filter of shape (C_out, C_in, K_h, K_w), and output of shape (C_out, H_out, W_out).

The transformation works as follows:

1. For each output spatial position (h, w), extract the input patch of shape (C_in, K_h, K_w) that the filter will be applied to.
2. Flatten this patch into a column vector of length C_in * K_h * K_w.
3. Stack all H_out * W_out such column vectors to form a matrix of shape (C_in * K_h * K_w, H_out * W_out).
4. Reshape the filter bank into a matrix of shape (C_out, C_in * K_h * K_w).
5. The convolution result is now a standard matrix multiplication: filter_matrix * input_matrix, producing a (C_out, H_out * W_out) output that is reshaped to (C_out, H_out, W_out).

The advantage of this approach is that the convolution is now a single large GEMM call, which can leverage highly optimized GEMM implementations on tensor cores or systolic arrays. The disadvantage is that im2col creates a larger intermediate matrix (the input data is replicated for each filter position), increasing memory footprint by a factor of approximately K_h * K_w.

Alternative approaches include implicit GEMM (which computes the index remapping on-the-fly without materializing the full im2col matrix), Winograd convolution (which reduces the number of multiplications at the cost of more additions and transformations), and FFT-based convolution (efficient for large filter sizes but rarely used in practice for the small 3x3 filters common in modern CNNs).

---

### Q3. What are the computational characteristics of the transformer attention mechanism?

**Answer:**

The multi-head self-attention mechanism in a transformer consists of several distinct compute phases, each with different characteristics:

**Linear projections (Q, K, V)**: The input tensor X of shape (B, S, D) is projected to queries, keys, and values via three weight matrices of shape (D, D). These are standard GEMMs with shapes (B*S, D) * (D, D), producing outputs of shape (B*S, D). These are compute-bound for large batch sizes and model dimensions.

**Attention score computation (Q*K^T)**: For each attention head, Q and K have shape (B, H, S, D/H) where H is the number of heads. The attention scores are computed as Q * K^T, which is a batched GEMM of shape (B*H, S, D/H) * (B*H, D/H, S), producing (B*H, S, S). This has quadratic memory and compute cost in sequence length S.

**Softmax**: Applied row-wise to the S x S attention score matrix. This is a memory-bound reduction operation.

**Value aggregation (scores * V)**: Another batched GEMM of shape (B*H, S, S) * (B*H, S, D/H), producing (B*H, S, D/H).

**Output projection**: A final GEMM to project the concatenated heads back to dimension D.

The total compute for self-attention is approximately 8*B*S^2*D + 4*B*S*D^2 FLOPS per layer. The S^2 term becomes dominant for long sequences (S > D), while the D^2 term dominates for short sequences with large model dimensions.

For hardware design, the key challenge is that attention involves both large compute-bound GEMMs (projections) and memory-bound operations (softmax, masking) interspersed in a way that makes fusion critical for performance. FlashAttention addresses this by fusing the entire attention computation into a single kernel that tiles across the sequence dimension and keeps intermediate results in on-chip SRAM.

---

### Q4. How do CNN workload characteristics differ from transformer workloads?

**Answer:**

CNNs and transformers differ in several ways that impact accelerator design:

| Characteristic | CNNs | Transformers |
|---|---|---|
| Core operation | Convolution (as GEMM) | Matrix multiply (GEMM) |
| Data reuse | High (filter reuse across spatial positions) | Moderate (weight reuse across batch/sequence) |
| Memory access pattern | Regular, strided | Regular for GEMMs, irregular for attention masking |
| Arithmetic intensity | High for conv layers, moderate overall | High for large GEMMs, low for attention/softmax |
| Scaling bottleneck | Compute | Memory bandwidth and capacity |
| Sequence dependence | None (spatial) | Quadratic attention (O(S^2)) |
| Parameter count | Moderate (millions) | Very large (billions to trillions) |
| Activation memory | Proportional to spatial resolution | Proportional to S^2 for attention maps |
| Batch size sensitivity | Less sensitive | More sensitive (small batches reduce GEMM efficiency) |

For accelerator architects, these differences mean that a chip optimized purely for CNNs (which dominated in 2012-2017) may not perform optimally on transformer workloads. Specifically, transformers demand more memory bandwidth per FLOP, larger on-chip memory for KV caches and attention maps, and efficient support for variable-length sequences and batched GEMMs with non-standard shapes.

The industry trend is clearly toward transformer-optimized designs. NVIDIA's Hopper architecture introduced the Transformer Engine with FP8 support specifically for this workload. Google's TPU v4 and v5 were designed with transformer training and inference as primary targets.

---

### Q5. What is a GEMM and why is it the most important operation to optimize?

**Answer:**

GEMM (General Matrix Multiply) computes C = alpha * A * B + beta * C, where A is an M x K matrix, B is a K x N matrix, and C is an M x N matrix. In deep learning, alpha = 1 and beta = 0 (or beta = 1 for bias addition), and the operation reduces to C = A * B (+ bias).

GEMM is the most important operation to optimize because:

1. **Dominance**: GEMMs account for 70-90% of the total FLOPS in both CNN and transformer workloads. Every fully connected layer, every attention projection, every feed-forward layer, and (via im2col) every convolution is a GEMM.

2. **High arithmetic intensity**: A GEMM of dimensions M, K, N performs 2*M*K*N FLOPS while moving M*K + K*N + M*N elements. For large square matrices, arithmetic intensity scales as O(N), meaning GEMMs can be made compute-bound by increasing matrix dimensions. This allows high utilization of compute resources.

3. **Regular access patterns**: GEMM accesses memory in predictable, strided patterns that are amenable to prefetching, tiling, and vectorization. There are no data-dependent control flow decisions.

4. **Tileability**: GEMM decomposes naturally into independent sub-problems (tiles) that can be computed in parallel. This maps directly to parallel hardware: different tiles can be assigned to different SMs, systolic arrays, or processing elements.

5. **Mature optimization**: Decades of research in high-performance computing have produced highly optimized GEMM implementations. Libraries like cuBLAS, MKL, and hardware features like tensor cores are specifically designed to maximize GEMM throughput.

The key parameters that determine GEMM performance are the matrix dimensions (M, K, N), the data type (FP32, FP16, BF16, FP8, INT8), and the memory layout (row-major, column-major). Choosing appropriate tile sizes to match the hardware's compute units and on-chip memory capacity is the central optimization challenge.

---

### Q6. What are the compute characteristics of the feed-forward network (FFN) in a transformer?

**Answer:**

The feed-forward network in each transformer layer typically consists of two linear transformations with a nonlinear activation in between:

```
FFN(x) = W2 * activation(W1 * x + b1) + b2
```

where x has shape (B*S, D), W1 has shape (D, 4D) (the hidden dimension is conventionally 4x the model dimension), and W2 has shape (4D, D). Some architectures use gated variants (SwiGLU) where the FFN involves an additional weight matrix and a gating mechanism.

The compute characteristics are:

**First linear layer (W1 * x)**: A GEMM of shape (B*S, D) * (D, 4D) requiring 2 * B*S * D * 4D = 8*B*S*D^2 FLOPS. This is compute-bound for large batch sizes or long sequences.

**Activation**: Element-wise GELU or SiLU applied to B*S * 4D elements. This is memory-bound (arithmetic intensity of ~1 FLOP per element loaded).

**Second linear layer (W2 * output)**: A GEMM of shape (B*S, 4D) * (4D, D) requiring 8*B*S*D^2 FLOPS.

The total FFN compute per layer is approximately 16*B*S*D^2 FLOPS (or 24*B*S*D^2 for gated variants with three matrices). For a model with D = 4096, B*S = 4096, a single FFN layer requires approximately 16 * 4096 * 4096^2 = 1.1 * 10^12 FLOPS = 1.1 TFLOPS.

The FFN accounts for roughly two-thirds of the total compute in a standard transformer layer (the remaining third being attention). Because both FFN GEMMs have large, regular dimensions, they are the most efficiently accelerated portion of the transformer and typically achieve high utilization on tensor cores and systolic arrays.

---

### Q7. How does batch size affect the compute characteristics of inference workloads?

**Answer:**

Batch size has a profound effect on inference performance because it determines the dimensions of the GEMMs involved and therefore their arithmetic intensity.

For a single linear layer with weight matrix W of shape (D_in, D_out) and input batch of shape (B, D_in), the GEMM computes output = input * W^T with 2 * B * D_in * D_out FLOPS. The data moved is B * D_in + D_in * D_out + B * D_out elements.

With B = 1 (single-sample inference), the operation degenerates to a matrix-vector multiply. The weights (D_in * D_out elements) must be loaded from memory for just 2 * D_in * D_out FLOPS of compute, giving arithmetic intensity of approximately 2 FLOPS per element, or roughly 1 FLOP/byte for FP16. This is extremely memory-bound on any modern accelerator.

With B = 256, the same weights are reused across 256 inputs, and the arithmetic intensity increases to approximately 256 FLOPS per weight element, or roughly 128 FLOPS/byte for FP16. This is solidly compute-bound on most hardware.

This relationship explains several phenomena:
- **Throughput-optimized inference** uses large batch sizes to maximize compute utilization and FLOPS/$ efficiency.
- **Latency-optimized inference** (e.g., interactive chat) uses batch size 1 or small batches, making it memory-bandwidth-bound and poorly utilizing compute resources.
- **Continuous batching** and **dynamic batching** are techniques to increase effective batch size by grouping multiple concurrent requests.
- Inference-focused accelerators (like Groq's LPU) may prioritize memory bandwidth over raw compute to better serve latency-sensitive, small-batch workloads.

---

### Q8. What is the significance of the "ops/byte" ridge point on the roofline model for real workloads?

**Answer:**

The ridge point of a hardware platform is the arithmetic intensity at which the workload transitions from being memory-bound to compute-bound. It is calculated as peak compute throughput divided by peak memory bandwidth:

```
Ridge point = peak_FLOPS / peak_bandwidth_bytes_per_second
```

| Accelerator | Peak FP16 TFLOPS | HBM BW (TB/s) | Ridge Point (FLOPS/byte) |
|---|---|---|---|
| NVIDIA A100 | 312 | 2.0 | 156 |
| NVIDIA H100 | 990 | 3.35 | 295 |
| NVIDIA B200 | ~2250 | 8.0 | 281 |
| Google TPU v5e | ~197 | 1.6 | 123 |
| AMD MI300X | 1300 | 5.3 | 245 |

The trend shows that ridge points are generally increasing over time (except when bandwidth grows faster, as with B200's HBM3e), meaning more workloads become memory-bound on newer hardware. This has important implications:

1. Operators that were compute-bound on A100 may be memory-bound on H100, requiring re-optimization.
2. The value of operator fusion (which reduces memory traffic) increases with each generation.
3. Techniques like FlashAttention, which trade extra compute for reduced memory traffic, become more valuable as the ridge point increases.
4. Lower precision (FP8, INT8) doubles or quadruples the peak compute, raising the ridge point further and making even more operations memory-bound unless the data can also be stored in lower precision.

Understanding where specific operations fall relative to the ridge point is essential for predicting performance and prioritizing optimization efforts.

---

### Q9. How do you profile an AI workload to identify bottlenecks?

**Answer:**

Profiling an AI workload involves several layers of analysis:

**Kernel-level profiling**: Tools like NVIDIA Nsight Compute (ncu) analyze individual GPU kernels, showing achieved FLOPS, memory bandwidth utilization, warp occupancy, stall reasons, and instruction mix. This reveals whether each kernel is compute-bound, memory-bound, or latency-bound (limited by pipeline stalls or synchronization).

**Trace-level profiling**: Tools like NVIDIA Nsight Systems (nsys), PyTorch Profiler, or TensorBoard trace viewers show the end-to-end execution timeline, revealing time spent in compute, memory transfers (host-to-device, device-to-host), communication (NCCL all-reduce), and idle gaps between operations. This identifies macro-level bottlenecks such as data loading stalls, CPU-side overhead, or communication not overlapping with compute.

**Roofline analysis**: Plotting each kernel's achieved performance against its arithmetic intensity on the roofline chart reveals how close each kernel is to its theoretical ceiling and whether the bottleneck is compute or bandwidth.

**Communication profiling**: For distributed training, profiling all-reduce, all-gather, and point-to-point communication reveals whether scaling is limited by interconnect bandwidth, latency, or software overhead.

A systematic profiling workflow is:
1. Capture a trace of one or more training/inference steps.
2. Identify the kernels or operations that consume the most wall-clock time.
3. For each top kernel, determine whether it is compute-bound or memory-bound using roofline analysis.
4. For compute-bound kernels, investigate utilization efficiency (occupancy, instruction-level parallelism).
5. For memory-bound kernels, investigate data reuse patterns and fusion opportunities.
6. For communication, investigate overlap with compute and bandwidth utilization.

---

### Q10. What are scaling laws and how do they relate to accelerator demand?

**Answer:**

Scaling laws, formalized by Kaplan et al. (2020) and refined by Hoffmann et al. (Chinchilla, 2022), describe the predictable relationship between model performance (measured as loss on a test set) and three variables: number of model parameters (N), training dataset size (D), and total training compute (C).

The key finding is that loss decreases as a power law with increasing compute:

```
L(C) = a * C^(-alpha)
```

where alpha is approximately 0.05-0.1, meaning that each 10x increase in compute reduces loss by a small but consistent and valuable amount. This power-law relationship has held across multiple orders of magnitude of compute, from small models trained on laptops to the largest frontier models trained on tens of thousands of GPUs.

The implications for accelerator demand are profound. Because the relationship is a power law rather than a plateau, there is no obvious point of diminishing returns where additional compute stops being useful. Each generation of larger models has delivered measurably better capabilities (better reasoning, broader knowledge, more natural language understanding), creating strong economic incentives to continue scaling.

The compute required for frontier model training has been growing at approximately 4-5x per year. This means that even with 2-3x generational improvements in accelerator throughput, the number of accelerators required grows by roughly 1.5-2.5x per year. This relentless growth in compute demand is what has made AI accelerator design one of the most active and economically important areas of computer architecture.
