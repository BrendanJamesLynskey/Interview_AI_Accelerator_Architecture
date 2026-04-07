# Quiz: Foundations

Test your knowledge of AI accelerator fundamentals, compute metrics, and workload characteristics.

---

**1. What primarily caused the end of Dennard scaling around 2006?**
- A) Transistors could no longer be made smaller
- B) Leakage current prevented further voltage scaling
- C) Clock frequencies reached the speed of light limit
- D) Manufacturing costs became prohibitive

**2. What is the arithmetic intensity of a matrix-vector multiply y = Wx where W is 4096x4096 in FP16 and x is a 4096-element vector?**
- A) 0.5 FLOPS/byte
- B) 1.0 FLOPS/byte
- C) 2.0 FLOPS/byte
- D) 4096 FLOPS/byte

**3. On the roofline model, the ridge point represents:**
- A) The maximum clock frequency of the processor
- B) The arithmetic intensity where the workload transitions from memory-bound to compute-bound
- C) The peak memory bandwidth in GB/s
- D) The thermal throttling threshold

**4. For a chip with 500 TFLOPS peak compute and 2 TB/s memory bandwidth, what is the ridge point?**
- A) 125 FLOPS/byte
- B) 250 FLOPS/byte
- C) 500 FLOPS/byte
- D) 1000 FLOPS/byte

**5. The 6ND approximation for training compute estimates total FLOPS as:**
- A) 6 * num_parameters * num_epochs
- B) 6 * num_parameters * num_tokens
- C) 6 * num_layers * num_tokens
- D) 6 * num_parameters * batch_size

**6. Which operation in a transformer has quadratic complexity in sequence length?**
- A) Feed-forward network
- B) Layer normalization
- C) Self-attention (Q*K^T)
- D) Embedding lookup

**7. A fused multiply-add (FMA) is counted as how many floating-point operations?**
- A) 1
- B) 2
- C) 3
- D) 4

**8. What percentage of a CPU's energy is typically spent on actual computation (vs control overhead)?**
- A) 50-60%
- B) 30-40%
- C) 15-25%
- D) 5-10%

**9. The im2col transformation converts convolution into:**
- A) A Fourier transform
- B) A general matrix multiply (GEMM)
- C) A sparse matrix operation
- D) A reduction operation

**10. Which metric is most important for data center AI accelerator evaluation?**
- A) Peak clock frequency
- B) Number of transistors
- C) Operations per watt (TOPS/W)
- D) Die area in mm^2

**11. The "hardware lottery" refers to:**
- A) The random nature of chip manufacturing defects
- B) Algorithms succeeding or failing based on available hardware compatibility
- C) The lottery system for allocating GPU access
- D) Random variation in chip performance

**12. At batch size 1, a fully connected layer during inference is best described as:**
- A) Compute-bound
- B) Memory-bandwidth-bound
- C) Latency-bound
- D) I/O-bound

**13. If compute throughput grows at 2x per generation and memory bandwidth at 1.5x, the ridge point:**
- A) Stays constant
- B) Decreases
- C) Increases
- D) Oscillates

**14. The scaling laws for LLMs show that loss decreases as a function of compute following:**
- A) An exponential decay
- B) A linear decrease
- C) A power law
- D) A step function

---

## Answer Key

1. **B** -- Leakage current from quantum tunneling prevented further voltage scaling, ending the constant power density that Dennard scaling predicted.

2. **B** -- FLOPS = 2 * 4096 * 4096 = 33.6M. Bytes = W (4096*4096*2 = 32MB) + x (4096*2 = 8KB) + y (4096*2 = 8KB) ≈ 32MB. AI = 33.6M / 32M ≈ 1.05 FLOPS/byte.

3. **B** -- The ridge point is where the bandwidth ceiling intersects the compute ceiling, marking the transition between memory-bound and compute-bound regimes.

4. **B** -- Ridge point = 500 TFLOPS / 2 TB/s = 250 FLOPS/byte.

5. **B** -- C = 6 * N * D where N is number of parameters and D is number of training tokens.

6. **C** -- Self-attention computes Q*K^T which produces an S*S matrix, giving O(S^2) complexity.

7. **B** -- An FMA computes a*b+c, which counts as one multiply (1 FLOP) and one add (1 FLOP) = 2 FLOPS.

8. **D** -- CPUs spend approximately 5-10% of energy on computation; the rest goes to instruction fetch/decode, branch prediction, caching, and other control overhead.

9. **B** -- im2col rearranges the input patches into columns of a matrix, transforming convolution into a GEMM.

10. **C** -- Operations per watt (TOPS/W or FLOPS/W) is the key efficiency metric because power is the binding constraint in data centers.

11. **B** -- The hardware lottery describes how algorithms that happen to map well onto existing hardware succeed regardless of theoretical merit.

12. **B** -- At batch size 1, the weight matrix must be fully loaded from memory for a single vector multiply, making the operation memory-bandwidth-bound.

13. **C** -- Ridge point = compute/bandwidth. If compute grows faster than bandwidth, the ridge point increases, making more workloads memory-bound.

14. **C** -- Scaling laws show loss decreasing as a power law with compute: L(C) = a * C^(-alpha).
