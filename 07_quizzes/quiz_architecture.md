# Quiz: Architecture

Test your knowledge of GPU architecture, systolic arrays, TPUs, and dataflow architectures.

---

**1. How many threads are in a single NVIDIA GPU warp?**
- A) 16
- B) 32
- C) 64
- D) 128

**2. What is the primary advantage of a systolic array over a GPU for matrix multiplication?**
- A) Higher clock frequency
- B) More flexible programming model
- C) Higher energy efficiency due to nearest-neighbor data movement
- D) Larger memory capacity

**3. In a weight-stationary dataflow, which data type is held fixed in the processing elements?**
- A) Input activations
- B) Output activations / partial sums
- C) Weights
- D) Gradients

**4. Google's bfloat16 format has the same dynamic range as which format?**
- A) FP16
- B) FP32
- C) FP64
- D) INT16

**5. The NVIDIA H100 Transformer Engine primarily enables efficient use of which data type?**
- A) FP32
- B) FP16
- C) FP8
- D) INT4

**6. What interconnect topology does the TPU v4 pod use?**
- A) 2D mesh
- B) Fat tree
- C) 3D torus
- D) Full crossbar

**7. NVIDIA's NVLink 4.0 (H100) provides what bidirectional bandwidth per GPU?**
- A) 300 GB/s
- B) 600 GB/s
- C) 900 GB/s
- D) 1800 GB/s

**8. A 256x256 systolic array with K-dimension tile of 256 achieves what steady-state utilization?**
- A) 25%
- B) 33%
- C) 67%
- D) Greater than 90% for large outer tile dimensions

**9. What is the key innovation of the Cerebras Wafer-Scale Engine?**
- A) Using HBM4 memory
- B) Building an entire accelerator on a single silicon wafer
- C) Using photonic interconnects
- D) Supporting FP64 tensor operations

**10. Groq's LPU achieves low latency primarily through:**
- A) Higher clock frequency than GPUs
- B) Larger HBM capacity
- C) Deterministic, compiler-scheduled execution eliminating runtime scheduling
- D) Using quantum computing principles

**11. In GPU architecture, occupancy refers to:**
- A) The percentage of transistors actively switching
- B) The ratio of active warps to maximum possible warps per SM
- C) The fraction of HBM capacity in use
- D) The percentage of tensor cores performing computation

**12. The Eyeriss architecture introduced which dataflow strategy?**
- A) Weight-stationary
- B) Output-stationary
- C) Row-stationary
- D) Input-stationary

**13. Which of these is NOT a characteristic of the NVIDIA A100?**
- A) 2:4 structured sparsity support
- B) FP8 tensor core operations
- C) 80 GB HBM2e
- D) 3rd-generation tensor cores

**14. The TPU v1 was designed exclusively for:**
- A) Training large language models
- B) Inference of quantized (INT8) models
- C) Scientific computing
- D) Graphics rendering

**15. What programming framework is most commonly used with Google TPUs?**
- A) PyTorch with CUDA
- B) JAX with XLA
- C) TensorFlow Lite
- D) OpenCL

**16. A key disadvantage of systolic arrays compared to GPUs is:**
- A) Lower peak throughput
- B) Higher power consumption
- C) Poor utilization for small or irregular matrix dimensions
- D) Inability to perform matrix multiplication

---

## Answer Key

1. **B** -- A warp contains 32 threads that execute in SIMT (Single Instruction, Multiple Threads) fashion.

2. **C** -- Systolic arrays achieve higher energy efficiency because data flows between nearest-neighbor PEs via short wires, minimizing data movement energy.

3. **C** -- In weight-stationary dataflow, weights are preloaded into PE registers and remain fixed while activations flow through.

4. **B** -- BF16 has 8 exponent bits (same as FP32), providing identical dynamic range (~10^38).

5. **C** -- The Transformer Engine enables FP8 computation with hardware-managed dynamic per-tensor scaling.

6. **C** -- TPU v4 pods use a 3D torus interconnect (ICI) for efficient collective communication.

7. **C** -- NVLink 4.0 provides 900 GB/s bidirectional bandwidth per GPU.

8. **D** -- With large outer tiles (M, N >> 256), the fill/drain overhead of the inner K=256 dimension is small, and utilization exceeds 90%. The formula is K/(K + 2*256 - 2) ≈ 256/510 = 50% per tile pass, but with large M, N the array stays in steady state most of the time.

9. **B** -- The WSE occupies an entire 300mm silicon wafer (~46,000 mm^2), containing 850,000 cores with 40 GB of distributed on-chip SRAM.

10. **C** -- The Groq LPU uses a compiler-scheduled (TISA) architecture where every operation is deterministically scheduled at compile time, eliminating cache misses and dynamic scheduling overhead.

11. **B** -- Occupancy is the ratio of active warps to the maximum warps an SM can support, indicating the potential for latency hiding through warp switching.

12. **C** -- Eyeriss introduced the row-stationary dataflow, which maximizes data reuse for all three data types (weights, inputs, partial sums) simultaneously.

13. **B** -- FP8 tensor core operations were introduced with the H100 (Hopper), not the A100 (Ampere). A100 introduced TF32 and 2:4 sparsity.

14. **B** -- The TPU v1 was an inference-only accelerator with a 256x256 INT8 systolic array, designed to serve Google's production inference workloads.

15. **B** -- JAX with XLA compilation is the primary framework for TPU development, though TensorFlow is also supported.

16. **C** -- Systolic arrays achieve high utilization only when matrix dimensions are multiples of the array size. Small or irregular dimensions waste processing elements.
