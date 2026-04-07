# Compute Fundamentals

This section covers the key metrics and models used to reason about AI accelerator performance, including FLOPS, TOPS, operations per watt, arithmetic intensity, and the roofline model.

---

### Q1. What is the difference between FLOPS, TFLOPS, and TOPS?

**Answer:**

FLOPS stands for Floating-Point Operations Per Second and measures the rate at which a processor can perform floating-point arithmetic (addition, subtraction, multiplication, division, and fused multiply-add). It is the standard throughput metric for scientific computing and AI training workloads. Prefix multipliers indicate scale: GFLOPS (10^9), TFLOPS (10^12), PFLOPS (10^15), and EFLOPS (10^18).

TOPS stands for Tera Operations Per Second and is typically used for integer operations, particularly INT8 or INT4 operations common in inference workloads. Because integer operations are simpler (smaller multipliers, no exponent handling), a chip can usually deliver higher TOPS than TFLOPS for the same silicon area and power.

It is critical to compare like with like. A chip advertising 2000 TOPS of INT8 is not directly comparable to one advertising 500 TFLOPS of FP16, because the operations differ in precision, dynamic range, and silicon cost. Furthermore, peak throughput assumes 100% utilization of all compute units, which is rarely achieved in practice. Sustained or "real-world" throughput on representative benchmarks (such as MLPerf) is a more meaningful comparison.

A fused multiply-add (FMA) is counted as two floating-point operations (one multiply and one add), which is why tensor core throughput figures are often double what you might expect from the number of MAC units multiplied by the clock frequency.

---

### Q2. What is arithmetic intensity and why is it important?

**Answer:**

Arithmetic intensity (also called operational intensity) is the ratio of compute operations to bytes of data moved, typically expressed in FLOPS/byte or ops/byte. It characterizes a workload's compute-to-communication ratio and determines whether that workload will be compute-bound or memory-bound on a given piece of hardware.

For example, consider a matrix multiplication C = A * B where A is M x K and B is K x N. The computation requires 2*M*K*N FLOPs (one multiply and one add per output element per inner dimension step). The minimum data movement is (M*K + K*N + M*N) elements read/written. For large square matrices (M = K = N), the arithmetic intensity approaches 2*N/3 FLOPS/byte (assuming each element is read once from memory), which grows with matrix size. This is why large GEMMs are compute-bound: they have high arithmetic intensity.

In contrast, element-wise operations like ReLU or layer normalization perform O(1) operations per element loaded from memory, giving an arithmetic intensity of roughly 0.25-1.0 FLOPS/byte depending on data type. These operations are memory-bound on any modern accelerator.

Understanding arithmetic intensity is essential for predicting bottlenecks, choosing appropriate hardware, and deciding where to invest optimization effort. Operations with low arithmetic intensity benefit from higher memory bandwidth, while those with high arithmetic intensity benefit from more compute units. See also the [roofline model](#q3-what-is-the-roofline-model-and-how-is-it-used).

---

### Q3. What is the roofline model and how is it used?

**Answer:**

The roofline model, introduced by Samuel Williams, Andrew Waterman, and David Patterson in 2009, is a visual performance model that plots achievable performance (in FLOPS) as a function of arithmetic intensity (in FLOPS/byte). The model defines two "rooflines" or performance ceilings:

1. **Compute ceiling**: A horizontal line at the hardware's peak compute throughput (e.g., 990 TFLOPS for H100 FP16 tensor cores).
2. **Bandwidth ceiling**: A diagonal line with slope equal to the peak memory bandwidth (e.g., 3.35 TB/s for H100 HBM3). The achievable performance for a bandwidth-bound kernel is arithmetic_intensity * bandwidth.

The intersection of these two lines occurs at the "ridge point," where arithmetic_intensity_ridge = peak_compute / peak_bandwidth. For the H100, this is approximately 990 TFLOPS / 3.35 TB/s = 295 FLOPS/byte.

Any kernel with arithmetic intensity below the ridge point is memory-bound (performance limited by bandwidth), and any kernel above is compute-bound (performance limited by peak compute). In practice, most real kernels fall below both ceilings due to inefficiencies such as cache misses, pipeline bubbles, memory bank conflicts, and suboptimal tiling.

The roofline model is used to: (a) identify the bottleneck for a given kernel on a given chip, (b) estimate the maximum achievable performance, (c) evaluate how close actual performance is to the theoretical limit, and (d) compare different hardware platforms. Extended versions of the model incorporate additional ceilings for on-chip SRAM bandwidth, network bandwidth, and instruction-level bottlenecks.

---

### Q4. What is ops/watt and why is it the key efficiency metric for data center accelerators?

**Answer:**

Ops/watt (operations per watt, typically expressed as TFLOPS/W or TOPS/W) measures the energy efficiency of a processor -- how much useful computation it can deliver per unit of power consumed. It has become the single most important metric for data center AI accelerators for several reasons.

First, power is the binding constraint in modern data centers. A typical data center rack can support 30-50 kW of power, and liquid-cooled AI racks push this to 70-120 kW. The total power budget of a data center is fixed by its electrical infrastructure and cooling capacity. Therefore, the total compute available is directly proportional to ops/watt of the deployed hardware.

Second, electricity is the largest operating cost for running AI workloads at scale. Training a large language model can cost millions of dollars in electricity alone. Improving ops/watt directly reduces the cost per trained model and the cost per inference query.

Third, thermal constraints are tightly coupled to power. Every watt consumed becomes a watt of heat that must be removed. Higher power densities require more expensive cooling solutions (liquid cooling, immersion cooling) and reduce chip reliability through thermal stress.

As a reference point, the NVIDIA H100 delivers approximately 990 TFLOPS of FP16 at 700W TDP, yielding roughly 1.4 TFLOPS/W. The Google TPU v5e, optimized for inference, achieves competitive TOPS/W figures at lower absolute power. Edge accelerators like Apple's Neural Engine achieve high TOPS/W at much lower absolute throughput, reflecting the different power budgets of mobile devices (5-15W) versus data center chips (300-700W).

---

### Q5. What is the difference between peak throughput and sustained throughput?

**Answer:**

Peak throughput is the theoretical maximum compute rate calculated by multiplying the number of compute units by the operations per unit per cycle by the clock frequency. For example, if a chip has 16,896 CUDA cores running at 1.98 GHz, each performing 2 FLOPS/cycle (one FMA), the peak FP32 throughput is 16,896 * 2 * 1.98 GHz = 66.9 TFLOPS.

Sustained throughput is the actual compute rate achieved on a real workload over a sustained period. It is always lower than peak throughput for several reasons:

**Memory stalls**: Compute units idle while waiting for data from memory or caches. This is the dominant cause of low utilization for memory-bound kernels.

**Pipeline bubbles**: Startup and drain phases at the beginning and end of computation, as well as dependencies between operations, leave compute units idle.

**Load imbalance**: In parallel workloads, some compute units may finish their work before others and sit idle.

**Control overhead**: Instruction fetch/decode, address calculation, and synchronization consume cycles that do not contribute to useful compute.

**Precision mismatch**: If a workload uses FP32 but the chip's peak throughput is quoted for FP16 tensor cores, the relevant peak is much lower.

On well-optimized GEMM workloads, modern GPUs can achieve 70-85% of peak tensor core throughput. On end-to-end model training, overall utilization (including communication, data loading, and non-GEMM operations) is typically 30-60%. The MLPerf benchmark suite provides standardized measurements of sustained throughput across different hardware platforms and workloads.

---

### Q6. What does it mean for a workload to be compute-bound vs memory-bound?

**Answer:**

A workload is compute-bound when the performance bottleneck is the rate at which the hardware can execute arithmetic operations. Adding more memory bandwidth would not improve performance, but adding more compute units (or increasing clock frequency) would. Compute-bound workloads have high arithmetic intensity, meaning they perform many operations per byte of data moved. Large matrix multiplications with dimensions in the thousands are typically compute-bound.

A workload is memory-bound when the performance bottleneck is the rate at which data can be moved between memory and compute units. The compute units are often idle, waiting for data. Adding more compute units would not help, but increasing memory bandwidth would. Memory-bound workloads have low arithmetic intensity. Examples include element-wise operations (activation functions, normalization), embedding table lookups, attention score computation for small batch sizes, and reduction operations.

In practice, a single neural network contains both compute-bound and memory-bound layers. The GEMM operations in fully connected and convolutional layers tend to be compute-bound (especially with large batch sizes), while activation functions, batch normalization, softmax, and data reshaping operations tend to be memory-bound. The overall performance of the network depends on how well the accelerator handles both types of workloads.

This distinction drives many architectural decisions: the ratio of compute units to memory bandwidth, the size and organization of on-chip SRAM (which can reduce off-chip memory accesses for memory-bound operations), and the software stack's ability to fuse multiple memory-bound operations into a single kernel that keeps data on-chip.

---

### Q7. How do you calculate the theoretical peak throughput of a tensor core array?

**Answer:**

The calculation follows a straightforward formula:

```
Peak throughput = num_tensor_cores * ops_per_core_per_cycle * clock_frequency
```

For the NVIDIA H100 SXM:
- 528 tensor cores (132 SMs * 4 tensor cores per SM)
- Each tensor core performs a 16x8x16 FP16 matrix multiply-accumulate per cycle, producing 16*8*16 = 2048 FMA operations = 4096 FLOPS (since each FMA counts as 2 FLOPS)
- Boost clock of approximately 1.83 GHz (though this varies with power and thermal conditions)

However, the published peak of approximately 990 TFLOPS for FP16 on H100 is calculated based on the actual tensor core throughput specifications provided by NVIDIA, which account for the specific microarchitectural details of the 4th-generation tensor cores.

For integer operations, the calculation is similar but uses the INT8 or INT4 multiply-accumulate throughput per tensor core, which is typically 2x or 4x the FP16 throughput respectively (because narrower data types allow more operations per cycle from the same hardware).

It is important to note that tensor core throughput requires data to be laid out in specific formats (fragments) and dimensions to be multiples of certain tile sizes. If the actual matrix dimensions do not align with these requirements, padding is needed, which reduces effective throughput. The software stack (cuBLAS, cuDNN) handles this transparently but the inefficiency still exists.

---

### Q8. What is the FMA (fused multiply-add) operation and why is it central to AI compute?

**Answer:**

A fused multiply-add (FMA) computes `a * b + c` in a single operation, using a single rounding step at the end rather than rounding after the multiply and again after the add. This is the fundamental operation in AI compute because the core computation in neural networks is the dot product: each output activation is the sum of products of inputs and weights, which is exactly a sequence of multiply-accumulate operations.

The FMA is central for several reasons. First, it provides two useful FLOPS per operation (one multiply and one add), which is why hardware vendors count each FMA as 2 FLOPS when reporting throughput. Second, the single rounding step provides slightly better numerical accuracy than performing the multiply and add separately, which matters for maintaining training stability in reduced-precision formats. Third, it maps directly to the inner loop of matrix multiplication and convolution, which dominate the compute time of neural network training and inference.

Hardware implementations of FMA units vary by precision. An FP32 FMA requires a 24-bit significand multiplier (producing a 48-bit product) and a 48-bit adder with alignment shift. An FP16 FMA uses an 11-bit multiplier and is roughly 4x cheaper in area and energy. An INT8 multiply-accumulate uses an 8-bit integer multiplier and is even cheaper. This cost differential is why reduced-precision compute (FP16, BF16, FP8, INT8) delivers dramatically higher throughput per unit area and power.

---

### Q9. How do you estimate the compute requirements for training a large language model?

**Answer:**

A widely used approximation from the Chinchilla scaling analysis estimates the total training compute as:

```
C = 6 * N * D
```

where C is total floating-point operations, N is the number of model parameters, and D is the number of training tokens. The factor of 6 accounts for the forward pass (2 FLOPS per parameter per token for the multiply-adds in each linear layer) plus the backward pass (approximately 4 FLOPS per parameter per token, since both activation gradients and weight gradients must be computed).

For example, a 70-billion parameter model trained on 2 trillion tokens requires approximately:
```
C = 6 * 70 * 10^9 * 2 * 10^12 = 8.4 * 10^23 FLOPS
```

On an H100 delivering 500 TFLOPS of sustained FP16 throughput (roughly 50% utilization of 990 TFLOPS peak), training would take:
```
T = 8.4 * 10^23 / (500 * 10^12) = 1.68 * 10^9 seconds = ~53 GPU-years
```

With 1024 GPUs, this becomes approximately 19 days, which aligns roughly with reported training times for models of this scale.

This estimate captures only the compute for linear layers (GEMMs). Attention score computation, normalization, activation functions, and communication overhead add approximately 10-30% to the total. Nevertheless, the 6ND approximation is a valuable back-of-envelope tool for estimating training cost and comparing hardware platforms.

---

### Q10. What is the compute-to-communication ratio and why does it matter for distributed training?

**Answer:**

The compute-to-communication ratio is the amount of useful computation performed between communication phases in a distributed training setup. In data-parallel training, each accelerator computes gradients on its local batch and then must synchronize gradients across all accelerators via an all-reduce operation. The compute time is proportional to the batch size per accelerator and the model size, while the communication time is proportional to the model size and inversely proportional to the interconnect bandwidth.

If the compute-to-communication ratio is low (meaning communication takes a significant fraction of total time), scaling to more accelerators yields diminishing returns. The system spends an increasing fraction of time communicating rather than computing. This is the fundamental scalability challenge of distributed training.

For a model with P parameters using FP16 (2 bytes per parameter) on N accelerators connected with bandwidth B per link using a ring all-reduce:
```
Communication time = 2 * (N-1)/N * P * 2 / B
Compute time = 6 * P * tokens_per_gpu / throughput_per_gpu
```

The ratio improves with: (a) larger batch sizes per GPU (more compute between communication phases), (b) higher interconnect bandwidth (NVLink at 900 GB/s vs PCIe at 64 GB/s), (c) computation-communication overlap (pipelining gradient all-reduce with backward pass computation), and (d) reduced communication volume (gradient compression, mixed-precision communication).

Understanding this ratio is essential for predicting the scalability of a training run and for making architectural decisions about interconnect bandwidth provisioning.
