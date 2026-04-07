# Worked Problem: Roofline Analysis

## Problem Statement

You are evaluating a neural network inference workload on two accelerator platforms. The workload consists of three kernel types:

| Kernel | FLOPS per invocation | Bytes accessed (off-chip) | Invocations per inference |
|---|---|---|---|
| GEMM (FC layers) | 4.0 * 10^9 | 32 * 10^6 | 12 |
| Softmax | 0.5 * 10^6 | 2.0 * 10^6 | 12 |
| Layer Norm | 1.0 * 10^6 | 4.0 * 10^6 | 24 |

The two platforms are:

| Platform | Peak FP16 Compute (TFLOPS) | HBM Bandwidth (TB/s) |
|---|---|---|
| Platform A | 300 | 2.0 |
| Platform B | 150 | 3.0 |

For each kernel on each platform:
1. Calculate the arithmetic intensity.
2. Determine whether the kernel is compute-bound or memory-bound.
3. Calculate the achievable throughput.
4. Estimate the total inference latency on each platform.

---

## Worked Solution

### Step 1: Calculate the ridge point for each platform

The ridge point is where the memory bandwidth ceiling intersects the compute ceiling:

```
Ridge point = Peak Compute / Peak Bandwidth
```

**Platform A**: 300 * 10^12 / (2.0 * 10^12) = 150 FLOPS/byte

**Platform B**: 150 * 10^12 / (3.0 * 10^12) = 50 FLOPS/byte

### Step 2: Calculate arithmetic intensity for each kernel

```
Arithmetic Intensity = FLOPS / Bytes accessed
```

**GEMM**: 4.0 * 10^9 / 32 * 10^6 = 125 FLOPS/byte

**Softmax**: 0.5 * 10^6 / 2.0 * 10^6 = 0.25 FLOPS/byte

**Layer Norm**: 1.0 * 10^6 / 4.0 * 10^6 = 0.25 FLOPS/byte

### Step 3: Classify each kernel on each platform

A kernel is memory-bound if its arithmetic intensity is below the ridge point, and compute-bound if above.

| Kernel | AI (FLOPS/byte) | Platform A (ridge=150) | Platform B (ridge=50) |
|---|---|---|---|
| GEMM | 125 | Memory-bound | Compute-bound |
| Softmax | 0.25 | Memory-bound | Memory-bound |
| Layer Norm | 0.25 | Memory-bound | Memory-bound |

Key observation: The GEMM kernel is memory-bound on Platform A (AI of 125 < ridge of 150) but compute-bound on Platform B (AI of 125 > ridge of 50). This illustrates how the same kernel can have different bottlenecks on different hardware.

### Step 4: Calculate achievable throughput for each kernel on each platform

For memory-bound kernels: Achievable TFLOPS = Arithmetic Intensity * Bandwidth
For compute-bound kernels: Achievable TFLOPS = Peak Compute

**Platform A:**
- GEMM: 125 * 2.0 = 250 TFLOPS (memory-bound, limited below the 300 TFLOPS peak)
- Softmax: 0.25 * 2.0 = 0.5 TFLOPS
- Layer Norm: 0.25 * 2.0 = 0.5 TFLOPS

**Platform B:**
- GEMM: 150 TFLOPS (compute-bound, at peak)
- Softmax: 0.25 * 3.0 = 0.75 TFLOPS
- Layer Norm: 0.25 * 3.0 = 0.75 TFLOPS

### Step 5: Calculate per-invocation latency for each kernel

```
Latency = FLOPS / Achievable throughput
```

**Platform A:**
- GEMM: 4.0 * 10^9 / (250 * 10^12) = 16.0 microseconds
- Softmax: 0.5 * 10^6 / (0.5 * 10^12) = 1.0 microseconds
- Layer Norm: 1.0 * 10^6 / (0.5 * 10^12) = 2.0 microseconds

**Platform B:**
- GEMM: 4.0 * 10^9 / (150 * 10^12) = 26.7 microseconds
- Softmax: 0.5 * 10^6 / (0.75 * 10^12) = 0.67 microseconds
- Layer Norm: 1.0 * 10^6 / (0.75 * 10^12) = 1.33 microseconds

### Step 6: Calculate total inference latency

```
Total = sum(per_kernel_latency * invocations)
```

**Platform A:**
- GEMMs: 16.0 us * 12 = 192.0 us
- Softmax: 1.0 us * 12 = 12.0 us
- Layer Norm: 2.0 us * 24 = 48.0 us
- **Total: 252.0 microseconds**

**Platform B:**
- GEMMs: 26.7 us * 12 = 320.0 us
- Softmax: 0.67 us * 12 = 8.0 us
- Layer Norm: 1.33 us * 24 = 32.0 us
- **Total: 360.0 microseconds**

### Step 7: Analysis and conclusions

Platform A (higher compute, lower bandwidth) is 1.43x faster overall despite the GEMM being memory-bound, because the GEMM kernel dominates total runtime (76% of time on Platform A) and achieves 250 TFLOPS on Platform A vs 150 TFLOPS on Platform B.

However, if operator fusion were used to eliminate the off-chip memory accesses for Softmax and Layer Norm (fusing them with adjacent GEMMs so data stays on-chip), the memory-bound kernels would effectively disappear, and Platform A's advantage would increase further.

This analysis demonstrates why roofline modeling is essential: Platform B has 2x the bandwidth but only half the compute, making it the worse choice for this GEMM-dominated workload despite its lower ridge point.
