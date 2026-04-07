# On-Chip Memory Design

This section covers the design of on-chip SRAM for AI accelerators, including SRAM bank organization, multi-ported memories, double buffering, and scratchpad design considerations.

---

### Q1. Why is on-chip SRAM critical for AI accelerator performance?

**Answer:**

On-chip SRAM serves as a high-bandwidth, low-latency buffer between the compute units and the slower off-chip HBM. Its importance stems from the massive bandwidth gap between these two levels: on-chip SRAM can provide 10-100 TB/s of aggregate bandwidth (depending on the number of banks and ports), while HBM provides 2-8 TB/s. This 10-50x bandwidth advantage means that data reused from SRAM rather than refetched from HBM can be accessed 10-50x faster.

For AI workloads, the amount of on-chip SRAM directly determines the tile sizes that can be used for matrix multiplication, which in turn determines the arithmetic intensity achieved. Larger tiles enable more data reuse per HBM fetch, increasing arithmetic intensity and moving the workload further into the compute-bound regime on the roofline model.

The tradeoff is that SRAM is expensive in silicon area. SRAM bit cells are 50-100x larger than DRAM bit cells and cannot be stacked vertically like HBM. On a modern AI accelerator, SRAM may occupy 30-50% of the total die area. Increasing SRAM capacity means either a larger (more expensive) die or fewer compute units. This area tradeoff is one of the central design decisions in AI accelerator architecture.

Modern AI accelerators allocate substantial SRAM: the H100 has approximately 50 MB of L2 cache plus 256 KB of shared memory per SM (33 MB aggregate across 132 SMs), totaling approximately 83 MB. The Cerebras WSE-2 takes this to the extreme with 40 GB of distributed SRAM. Google's TPU v4 has substantial on-chip memory per core for weight staging and activation buffering.

---

### Q2. How are SRAM banks organized in an AI accelerator?

**Answer:**

SRAM in AI accelerators is organized into multiple banks that can be accessed independently and in parallel. Banking is essential for achieving the aggregate bandwidth needed to feed the compute units.

A typical organization for a shared scratchpad might be 32 banks, each 8 KB wide (for a 256 KB scratchpad), with each bank providing one read or write per cycle. If the clock is 1 GHz and each access is 128 bits (16 bytes), each bank provides 16 GB/s, and 32 banks provide 512 GB/s aggregate bandwidth.

Bank conflicts occur when two or more accesses in the same cycle target the same bank but different addresses. Since each bank has only one port (typically), the conflicting accesses must be serialized, reducing effective bandwidth. In GPU shared memory, bank conflicts are a well-known performance hazard: NVIDIA shared memory has 32 banks, and if all 32 threads in a warp access different addresses in the same bank, the accesses serialize into 32 separate transactions, reducing bandwidth by 32x.

Strategies to minimize bank conflicts include:
- **Padding**: Adding extra columns to the data layout so that stride-based access patterns hit different banks
- **Address interleaving**: Distributing consecutive addresses across banks so that strided accesses are spread
- **Bank-aware tiling**: Choosing tile sizes and access patterns that naturally avoid conflicts
- **Multi-ported SRAM**: Using dual-ported or multi-ported SRAM cells (at the cost of larger cell area)

For systolic arrays, the global buffer SRAM is often designed with enough banks that all PEs can read their required data simultaneously. A 256-PE array might have 256 SRAM banks, one per PE, with an address interleaving scheme that maps each PE's access to a different bank.

---

### Q3. What are the tradeoffs between shared memory (scratchpad) and cache for on-chip storage?

**Answer:**

On-chip memory can be implemented as either a software-managed scratchpad or a hardware-managed cache. Each approach has distinct tradeoffs:

**Scratchpad (explicitly managed):**
- The software (compiler or programmer) explicitly controls what data is loaded into the scratchpad, when it is loaded, and when it is evicted.
- Advantages: no cache miss overhead (every access is guaranteed to hit), no tag storage overhead (no tag comparison logic), deterministic latency, and the ability to perfectly optimize data placement for the specific access pattern.
- Disadvantages: requires compiler or programmer expertise to manage, less flexible for irregular access patterns, and the software must handle all data movement scheduling.

**Cache (implicitly managed):**
- Hardware automatically loads data on cache misses and evicts data based on a replacement policy (e.g., LRU).
- Advantages: transparent to software, handles irregular access patterns automatically, and adapts to runtime data access behavior.
- Disadvantages: cache miss overhead (stall cycles when data is not present), tag storage area overhead (typically 5-10% of data area), potential for thrashing and pathological eviction patterns, and non-deterministic latency.

AI accelerators predominantly use scratchpads (or configurable shared memory) rather than caches for their primary on-chip storage, because AI workloads have highly predictable, regular access patterns that are well-suited to explicit management. The compiler can determine exactly which tiles to load and when, eliminating the overhead and uncertainty of a cache.

NVIDIA GPUs offer a hybrid: the shared memory / L1 cache in each SM is configurable. The programmer can allocate more capacity to shared memory (scratchpad) for kernels that benefit from explicit management, or more to L1 cache for kernels with less predictable access patterns. This flexibility allows the same hardware to serve both AI workloads (scratchpad-heavy) and general-purpose workloads (cache-heavy).

---

### Q4. How does the shared memory size affect GEMM tiling and performance?

**Answer:**

The shared memory size directly constrains the tile sizes that can be used for GEMM, which in turn determines the arithmetic intensity and performance. Consider a tiled GEMM computing C[Tm x Tn] += A[Tm x Tk] * B[Tk x Tn]:

The shared memory must hold tiles of A, B, and optionally the partial result C:
```
SRAM required = (Tm * Tk + Tk * Tn) * bytes_per_element
```

(With double buffering, multiply by 2.)

For FP16 on an SM with 128 KB of shared memory (a typical H100 configuration):
```
128 * 1024 / 2 = 65,536 FP16 elements available
With double buffering: 32,768 elements per buffer
```

If Tk = 32 and Tm = Tn = T:
```
T * 32 + 32 * T = 64T elements per buffer
64T = 32,768 -> T = 512
```

So we could use 512 x 512 output tiles with Tk = 32. The arithmetic intensity of this tiling:
```
AI = 2 * 512 * 32 * 512 / ((512 * 32 + 32 * 512 + 512 * 512) * 2)
   = 16.78M / (557K * 2) = 16.78M / 1.11M = 15.1 FLOPS/byte
```

Wait, this counts the output tile write. Actually, the key metric is how many FLOPS per byte of HBM traffic. For one sweep through K:
```
HBM reads per tile = (Tm*Tk + Tk*Tn) * 2 bytes * (K/Tk) = (512*32 + 32*512) * 2 * (K/32)
Total FLOPS per tile = 2 * Tm * K * Tn = 2 * 512 * K * 512
AI = 2 * 512 * K * 512 / ((512*32 + 32*512) * 2 * (K/32) + 512*512*2)
```

For large K, the output write is negligible and:
```
AI ≈ 2 * 512 * 512 / (2 * (512 + 512) * 2) = 524288 / 4096 = 128 FLOPS/byte
```

Larger shared memory enables larger tiles and higher arithmetic intensity. If shared memory were 256 KB:
```
Tiles could be 724 x 724 with Tk=32 (approximately)
AI ≈ 2 * 724 * 724 / (2 * (724 + 724) * 2) = 181 FLOPS/byte
```

This demonstrates why SRAM capacity is so impactful: doubling shared memory can increase arithmetic intensity by roughly 40%, potentially converting a memory-bound kernel into a compute-bound one.

---

### Q5. What is a global buffer and how is it used in spatial architectures?

**Answer:**

A global buffer (also called a shared buffer or activation buffer) is a large on-chip SRAM that serves as the primary data staging area between off-chip memory (HBM/DRAM) and the processing element array in a spatial accelerator architecture.

In the Eyeriss architecture, the global buffer is a 108 KB SRAM that stores input activations, weights, and output activations. It acts as the intermediary between the DRAM interface and the 168-PE spatial array. The dataflow controller reads data from the global buffer and distributes it to the appropriate PEs according to the configured dataflow pattern.

Key design considerations for a global buffer:

**Bandwidth provisioning**: The global buffer must supply data to all PEs simultaneously. For a 256-PE array where each PE needs one activation and one weight per cycle at 1 GHz, the buffer must provide 256 * 2 * 1 byte = 512 bytes/cycle = 512 GB/s. This requires many parallel SRAM banks with careful interleaving.

**Capacity sizing**: The buffer must hold enough data to sustain computation for a meaningful period before requiring a refill from DRAM. For tiled matrix multiplication, the buffer holds the current tiles of both input matrices. Larger buffers enable larger tiles and higher data reuse.

**Multi-purpose storage**: The buffer typically stores activations (input and output), weights (for the current tile), and possibly partial sums. The allocation between these data types may be configurable or determined by the compiler.

**Interface to PE array**: Data distribution from the global buffer to the PEs requires a distribution network (multicast buses, crossbar, or NoC). The design of this network determines how flexibly different dataflow patterns can be supported.

The global buffer is distinct from the L2 cache in a GPU, though it serves a similar role. The key difference is that the global buffer is explicitly managed (the compiler/controller determines what data to load and when) while the L2 cache uses hardware-based caching policies.

---

### Q6. How do multi-ported memories work and when are they used?

**Answer:**

A multi-ported memory cell allows multiple simultaneous accesses (reads and/or writes) to different addresses. A dual-ported SRAM cell has two independent sets of bitlines and wordlines, allowing two accesses per cycle. More ports are possible but increasingly expensive.

The area cost of multi-porting is significant: a standard 6T SRAM cell occupies approximately 0.04 um^2 at 7nm. A dual-ported (8T) cell is roughly 1.5x larger, and a two-read-one-write (2R1W) cell using 8T is similarly 1.5-1.8x. A full dual-port cell (2R2W, 10T) is approximately 2x the area. Adding more ports increases area super-linearly.

In AI accelerators, multi-ported memories are used in specific contexts:

**Register files**: The register file in a GPU SM is multi-ported to allow multiple warp schedulers to read operands and write results simultaneously. A register file supporting 4 warps might have 8+ read ports and 4+ write ports.

**Accumulation buffers**: In systolic arrays, the output accumulation buffer may need simultaneous read (to read the current partial sum) and write (to update it), requiring at least 1R1W ports.

**PE local storage**: Each PE's local scratchpad may be dual-ported to allow simultaneous data input (from the interconnect) and data output (to the MAC unit).

For large shared memories (global buffers, L2 caches), multi-porting is prohibitively expensive. Instead, the memory is divided into many single-ported banks that can be accessed independently. With sufficient banks and proper address interleaving, the aggregate bandwidth approaches what a multi-ported memory would provide, without the per-cell area overhead. This banking approach is the standard solution in AI accelerators.

---

### Q7. What is the role of data layout (NCHW vs NHWC) in memory performance?

**Answer:**

Data layout refers to the ordering of tensor dimensions in memory. For a 4D tensor with dimensions Batch (N), Channels (C), Height (H), Width (W), the two primary layouts are:

**NCHW**: Dimensions are stored in the order N, C, H, W. Spatial positions within a channel are contiguous in memory. This was historically the default in many frameworks and is natural for spatial operations.

**NHWC**: Dimensions are stored in the order N, H, W, C. All channels at a given spatial position are contiguous. This is more natural for operations that reduce across channels (like 1x1 convolutions and fully connected layers).

The layout matters for memory performance because:

1. **Vectorized loads**: Hardware loads data in cache lines or wide memory transactions (e.g., 128 bytes). If the computation needs data along the channel dimension, NHWC layout provides contiguous channel data in a single load, while NCHW requires gathering from non-contiguous addresses.

2. **Tensor core alignment**: NVIDIA tensor cores expect specific data layouts. For matrix operations, the inner dimension (typically the channel or reduction dimension) should be contiguous for efficient loading into tensor core fragments. NHWC aligns with this for most convolution and GEMM operations.

3. **Convolution efficiency**: Modern convolution implementations (cuDNN) map convolutions to GEMMs where the channel dimension is the inner (K) dimension. NHWC makes the K dimension contiguous, enabling coalesced memory access.

NVIDIA recommends NHWC layout for AI workloads on modern GPUs, and cuDNN's fastest convolution algorithms require NHWC. TensorFlow uses NHWC by default; PyTorch historically used NCHW but now supports and often recommends "channels last" (NHWC) format.

Transposing data between layouts is expensive (it requires reading and writing every element) and should be minimized. The best practice is to use a consistent layout (NHWC) throughout the computation graph and only transpose at input/output boundaries if needed.

---

### Q8. How does activation checkpointing trade compute for memory?

**Answer:**

Activation checkpointing (also called gradient checkpointing or rematerialization) is a memory optimization technique for training that reduces activation memory usage at the cost of additional computation during the backward pass.

During the forward pass of a neural network, the activations at each layer must be saved for use during the backward pass (to compute gradients via the chain rule). For a model with L layers and activation size A per layer, this requires L * A bytes of memory, which can be enormous for deep models with large intermediate tensors.

Activation checkpointing works by saving activations only at selected "checkpoint" layers during the forward pass and recomputing the activations for non-checkpointed layers during the backward pass. When the backward pass reaches a non-checkpointed layer, it re-executes the forward pass from the nearest checkpoint to regenerate the needed activations.

The tradeoff:
- **Memory saved**: Instead of storing activations for all L layers, only sqrt(L) checkpoint layers' activations are stored (with optimal placement), reducing memory from O(L*A) to O(sqrt(L)*A).
- **Compute overhead**: The forward pass is approximately executed twice (once during the actual forward pass and once during recomputation in the backward pass), increasing total compute by approximately 33% (since the backward pass is already ~2x the forward pass).

For large language models with hundreds of layers and billions of parameters, activation checkpointing is essential for fitting training within available HBM capacity. Without it, the activation memory for a 175B parameter model with sequence length 2048 and batch size 1 would exceed 100 GB per GPU -- more than any single GPU's HBM capacity.

From a hardware perspective, activation checkpointing increases the compute-to-memory-access ratio of training, which means the workload shifts further toward compute-bound, actually improving compute unit utilization on hardware with limited memory bandwidth.

---

### Q9. What is the impact of memory alignment and coalescing on performance?

**Answer:**

Memory alignment and coalescing are critical for achieving peak memory bandwidth utilization on AI accelerators.

**Alignment** refers to the starting address of a data access relative to the memory system's natural boundaries. HBM and DRAM systems transfer data in fixed-size bursts (typically 32 or 64 bytes for HBM). If a data access is not aligned to a burst boundary, it may require two bursts instead of one, halving effective bandwidth. Ensuring that tensor base addresses and strides are aligned to burst boundaries is a basic optimization that should always be applied.

**Coalescing** is the hardware's ability to combine multiple small memory accesses from different threads into a single large memory transaction. On NVIDIA GPUs, when 32 threads in a warp each request a 4-byte word from consecutive addresses, the hardware coalesces these into a single 128-byte transaction. If the addresses are non-consecutive or spread across cache lines, multiple transactions are needed.

The impact of poor coalescing can be severe. If 32 threads each access a random 4-byte address, the worst case requires 32 separate 128-byte transactions (fetching 4096 bytes but using only 128), reducing effective bandwidth by 32x. In practice, the reduction is usually less extreme but still significant: 2-8x bandwidth waste is common for poorly optimized kernels.

For AI workloads, coalescing is ensured by:
- Using appropriate data layouts (NHWC for convolutions on GPUs)
- Ensuring that the innermost loop iterates over the contiguous dimension
- Padding tensors so that row strides are multiples of the transaction size
- Using vectorized load instructions (e.g., float4 loads that access 16 bytes per thread)

The compiler and runtime handle much of this automatically for standard operations (cuBLAS, cuDNN), but custom CUDA kernels and non-standard access patterns require explicit attention to alignment and coalescing.

---

### Q10. How do you decide the ratio of compute to on-chip SRAM in an accelerator design?

**Answer:**

The compute-to-SRAM ratio is a fundamental architectural decision that determines the accelerator's position on the roofline model. The decision is driven by the target workload's arithmetic intensity distribution.

**Analysis approach:**

1. Profile the target workloads (e.g., transformer training and inference at various batch sizes) to determine the distribution of arithmetic intensities across all kernels.

2. For each kernel, determine the SRAM capacity needed to achieve a tile size that makes the kernel compute-bound (i.e., arithmetic intensity above the ridge point).

3. The ridge point is: peak_compute / HBM_bandwidth. More SRAM does not change the ridge point (which is determined by HBM bandwidth), but it enables larger tiles that achieve higher arithmetic intensity.

4. Choose the SRAM capacity such that the most important kernels (by total runtime) are compute-bound. Diminishing returns set in once the dominant kernels are already compute-bound.

**Practical considerations:**

- **Area budget**: SRAM area = capacity / density. At 7nm, SRAM density is approximately 25 Mb/mm^2. A 50 MB L2 cache requires approximately 16 mm^2, which is significant on a 600-800 mm^2 die.
- **Bandwidth scaling**: More SRAM banks provide more aggregate bandwidth but also more routing complexity and area overhead.
- **Diminishing returns**: Doubling SRAM capacity increases achievable arithmetic intensity by roughly sqrt(2)x (because tile dimensions scale as sqrt(SRAM)). The benefit curve flattens as tiles become large enough to make all GEMMs compute-bound.
- **Non-GEMM operations**: Memory-bound operations (softmax, normalization) benefit from SRAM capacity only through fusion -- keeping intermediate results on-chip. The SRAM must be large enough to hold the intermediate tensors for fused kernel sequences.

As a rule of thumb, modern AI accelerators allocate 30-50% of die area to SRAM and memory systems, with the remainder split between compute units, interconnect, and I/O. The exact ratio varies: inference-focused chips may allocate more SRAM (to handle memory-bound small-batch inference), while training-focused chips may allocate more compute (since large-batch training is compute-bound).
