# Quiz: Memory and Compute

Test your knowledge of memory hierarchy, HBM, MAC units, quantization, and sparsity.

---

**1. HBM achieves high bandwidth primarily through:**
- A) Higher clock frequencies than DDR
- B) A very wide interface (1024 bits) enabled by through-silicon vias (TSVs)
- C) Using faster transistor technology
- D) Larger DRAM cells

**2. Accessing a byte from HBM costs approximately how much more energy than from a register?**
- A) 2x
- B) 20x
- C) 200x
- D) 2000x

**3. Double buffering requires what tradeoff?**
- A) 2x compute units for the same throughput
- B) 2x SRAM capacity but hides memory loading latency
- C) 2x HBM bandwidth
- D) 2x clock frequency

**4. An INT8 MAC unit is approximately what factor cheaper in area than an FP32 MAC?**
- A) 2x
- B) 4x
- C) 9-11x
- D) 32x

**5. The purpose of FP32 accumulation in FP16 tensor core operations is to:**
- A) Increase the clock frequency
- B) Prevent precision loss from accumulating many small products
- C) Reduce memory bandwidth
- D) Enable backward pass computation

**6. In NVIDIA's 2:4 structured sparsity, what fraction of elements in each group of 4 must be zero?**
- A) 1 out of 4
- B) 2 out of 4
- C) 3 out of 4
- D) Variable

**7. GPTQ achieves high-quality weight quantization by:**
- A) Training the model from scratch with quantized weights
- B) Using the Hessian to compensate remaining weights for each quantization error
- C) Simply rounding weights to the nearest integer
- D) Using a neural network to predict optimal quantization levels

**8. The KV cache size grows with which variable during autoregressive generation?**
- A) Model dimension
- B) Number of layers
- C) Sequence length (number of generated tokens)
- D) Batch size only

**9. HBM3 has how many independent channels per stack?**
- A) 4
- B) 8
- C) 16
- D) 32

**10. What is the primary benefit of operator fusion for memory-bound operations?**
- A) Reduces the number of floating-point operations
- B) Eliminates intermediate writes to and reads from off-chip memory
- C) Increases the clock frequency
- D) Enables lower precision computation

**11. SmoothQuant addresses the activation outlier problem by:**
- A) Clipping outlier values
- B) Using higher precision for outlier channels
- C) Mathematically migrating quantization difficulty from activations to weights via per-channel scaling
- D) Training the model to avoid outliers

**12. The silicon interposer in HBM packaging provides:**
- A) Additional compute logic
- B) High-density wiring between the processor die and HBM stacks
- C) Cooling channels
- D) Power regulation

**13. Tiling a GEMM with larger tiles results in:**
- A) Lower arithmetic intensity
- B) Higher arithmetic intensity (more reuse per HBM fetch)
- C) More HBM accesses
- D) Lower compute utilization

**14. Weight-only quantization (INT4 weights, FP16 compute) is popular for LLM inference because:**
- A) It improves model accuracy
- B) It directly reduces the dominant cost (weight loading from HBM) for bandwidth-bound inference
- C) INT4 computation is faster than FP16
- D) It reduces the number of model parameters

**15. Per-channel quantization provides better accuracy than per-tensor quantization because:**
- A) It uses more bits per weight
- B) Different channels have different value ranges, and per-channel scales capture this variation
- C) It eliminates the need for calibration
- D) It reduces the model size further

**16. FlashAttention achieves its speedup primarily by:**
- A) Using a faster attention algorithm with lower complexity
- B) Fusing the attention computation to avoid materializing the S*S attention matrix in HBM
- C) Using INT8 quantization for attention scores
- D) Pruning attention heads

---

## Answer Key

1. **B** -- HBM uses TSVs to connect stacked DRAM dies, providing a 1024-bit wide interface that is 16x wider than a DDR channel.

2. **C** -- HBM access costs approximately 200 pJ per byte vs ~1 pJ for register access, a ~200x difference.

3. **B** -- Double buffering uses 2x the SRAM (one buffer for current computation, one for loading the next data) but hides memory latency by overlapping load and compute.

4. **D** -- The significand arrays alone suggest (24/8)^2 = 9x, but a floating-point unit also needs exponent handling, alignment and normalisation. Horowitz's widely cited 45 nm figures (ISSCC 2014) give 282 um^2 for an 8-bit multiplier vs 7,700 um^2 for an FP32 multiplier (27x), and 36-137 um^2 for an 8-32-bit integer adder vs 4,184 um^2 for an FP32 adder. An INT8 MAC is therefore roughly 28-37x smaller than an FP32 MAC, closest to 32x.

5. **B** -- FP32 accumulation prevents "swamping" where large partial sums cause small products to be rounded to zero, which would lose information when accumulating thousands of products.

6. **B** -- 2:4 means exactly 2 out of every 4 consecutive elements are zero (50% sparsity).

7. **B** -- GPTQ uses the Hessian (second-order information) to optimally adjust remaining unquantized weights to compensate for each weight's quantization error.

8. **C** -- The KV cache stores key and value vectors for all previous tokens, so it grows linearly with sequence length.

9. **C** -- HBM3 has 16 independent 64-bit channels per stack (vs 8 channels of 128 bits in HBM2).

10. **B** -- Fusion keeps intermediate results in registers or shared memory, eliminating the HBM write-then-read pattern between sequential operations.

11. **C** -- SmoothQuant scales up weights in channels with large activation outliers (and inversely scales activations), transferring the quantization difficulty to the more uniform weight distribution.

12. **B** -- The silicon interposer provides fine-pitch wiring (1-10 um) to connect the 1024+ data pins between the processor and HBM stacks, which organic substrates cannot achieve.

13. **B** -- Larger tiles increase data reuse: more compute operations are performed on each byte loaded from HBM, increasing arithmetic intensity.

14. **B** -- At batch size 1, inference is entirely memory-bandwidth-bound. INT4 weights reduce the weight data by 4x, directly speeding up the dominant bottleneck (weight loading).

15. **B** -- Different output channels (rows of the weight matrix) can have very different value ranges. Per-channel scales capture this variation, using the quantization range efficiently for each channel.

16. **B** -- FlashAttention tiles the attention computation along the sequence dimension and keeps intermediate results (attention scores, softmax statistics) in on-chip SRAM, never materializing the full S*S matrix in HBM.
