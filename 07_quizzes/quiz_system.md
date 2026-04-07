# Quiz: System

Test your knowledge of interconnects, compilers, power management, and training/inference systems.

---

**1. In a ring all-reduce with N nodes, the total data communicated per node is:**
- A) data_size
- B) 2 * (N-1)/N * data_size
- C) N * data_size
- D) data_size / N

**2. A 2D torus has what advantage over a 2D mesh?**
- A) Fewer links per node
- B) Simpler routing algorithm
- C) Doubled bisection bandwidth and halved diameter
- D) Lower power consumption

**3. Tensor parallelism is typically used within a single node (not across nodes) because:**
- A) It requires more memory than pipeline parallelism
- B) It requires per-layer all-reduce communication, needing NVLink-class bandwidth
- C) It only works with 8 GPUs
- D) It does not support gradient computation

**4. The XLA compiler's primary advantage over per-kernel compilation is:**
- A) Faster compilation time
- B) Whole-program optimization including cross-operation fusion and scheduling
- C) Support for more programming languages
- D) Better debugging tools

**5. At 700W TDP, approximately what fraction of power goes to the MAC compute units?**
- A) 80-90%
- B) 60-70%
- C) 35-50%
- D) 10-20%

**6. Dynamic voltage and frequency scaling (DVFS) reduces power by:**
- A) Reducing the number of active transistors
- B) Lowering voltage (quadratic power reduction) and frequency (linear)
- C) Switching to a different process node
- D) Reducing memory bandwidth

**7. For a hierarchical all-reduce (intra-node then inter-node), the key benefit is:**
- A) Reduced total data movement
- B) The intra-node phase uses high-bandwidth NVLink, and the inter-node phase operates on reduced data
- C) Simpler implementation
- D) Better accuracy of the reduced values

**8. The prefill phase of LLM inference is characterized by:**
- A) Matrix-vector multiplication (memory-bound)
- B) Large GEMM operations processing all prompt tokens (compute-bound)
- C) KV cache reads (bandwidth-bound)
- D) Weight quantization (compute-bound)

**9. Continuous batching improves inference serving throughput by:**
- A) Using larger batch sizes
- B) Dynamically adding/removing requests from the batch at each generation step
- C) Running multiple model replicas
- D) Using lower precision

**10. Speculative decoding reduces latency by:**
- A) Using a faster model for all inference
- B) Draft model generates multiple candidate tokens verified in parallel by the target model
- C) Reducing model size through pruning
- D) Increasing batch size

**11. CXL (Compute Express Link) is relevant for AI because it enables:**
- A) Faster GPU clock speeds
- B) Coherent shared memory pools that extend effective memory beyond local HBM
- C) Wireless chip-to-chip communication
- D) Quantum computing integration

**12. The Triton programming language differs from CUDA primarily in:**
- A) Operating at a tile/block level of abstraction rather than individual threads
- B) Only supporting integer computation
- C) Being slower than CUDA
- D) Only running on AMD GPUs

**13. Power delivery for a 1000W chip drawing ~1300A requires:**
- A) A single-phase voltage regulator
- B) Standard PCIe power connectors only
- C) Multi-phase VRMs with careful PDN design to handle di/dt transients
- D) External power supply only

**14. Mixture of Experts (MoE) models present which unique hardware challenge?**
- A) They require FP64 computation
- B) Load imbalance across experts and all-to-all communication for token routing
- C) They cannot be quantized
- D) They only work on TPUs

**15. Pipeline parallelism introduces "bubbles" because:**
- A) Network packets are lost
- B) Pipeline stages must wait during the fill and drain phases when not all stages are active
- C) Memory leaks in the software
- D) Thermal throttling reduces clock frequency

**16. Which of these is NOT a benefit of liquid cooling for AI accelerators?**
- A) Higher heat removal capacity than air cooling
- B) Enables higher TDP chips
- C) Reduces chip leakage power
- D) Supports higher rack power densities

---

## Answer Key

1. **B** -- Ring all-reduce consists of reduce-scatter and all-gather phases, each communicating (N-1)/N * data_size per node, totaling 2*(N-1)/N * data_size.

2. **C** -- The wraparound links in a torus provide additional paths across the bisection (doubling bandwidth) and reduce the maximum hop count by half (halving diameter).

3. **B** -- Tensor parallelism requires an all-reduce of activation tensors after every layer, which demands high bandwidth. Only NVLink (900 GB/s) provides sufficient bandwidth; InfiniBand (50 GB/s) would create severe communication bottlenecks.

4. **B** -- XLA sees the entire computation graph, enabling global optimizations like cross-operation tiling, end-to-end buffer reuse, and computation-communication overlap scheduling.

5. **C** -- The MAC units typically consume 35-50% of total chip power. The remainder goes to data movement (SRAM, HBM, NoC), clock distribution, leakage, and I/O.

6. **B** -- Power scales as C*V^2*f. Reducing voltage (V) provides quadratic savings; reducing frequency (f) provides linear savings. Both together provide cubic savings for a given reduction factor.

7. **B** -- The hierarchical approach exploits the bandwidth hierarchy: the intra-node phase uses fast NVLink (900 GB/s) to reduce data locally, so the inter-node phase (on slower InfiniBand) transfers much less data.

8. **B** -- Prefill processes all prompt tokens in one forward pass, creating large GEMMs with high arithmetic intensity. This makes it compute-bound, unlike the decode phase which is memory-bound.

9. **B** -- Continuous batching dynamically manages the active request set, immediately replacing completed requests with new ones, maximizing GPU utilization and throughput.

10. **B** -- A small draft model generates K candidate tokens quickly, and the large target model verifies all K in a single forward pass (which costs about the same as generating 1 token), potentially accepting multiple tokens per step.

11. **B** -- CXL enables coherent access to disaggregated memory pools, extending effective memory capacity beyond local HBM for serving very large models.

12. **A** -- Triton programs express computation on blocks/tiles of data, and the compiler automatically handles thread mapping, shared memory management, and hardware-specific optimization.

13. **C** -- At ~1300A, multi-phase (12-16 phase) buck converters with extensive decoupling capacitors and carefully designed power distribution networks are essential to handle the current magnitude and fast transients.

14. **B** -- MoE models route different tokens to different experts, creating uneven load across compute units and requiring all-to-all communication to move tokens between devices hosting different experts.

15. **B** -- In pipeline parallelism, the first stage must fill the pipeline before all stages are active, and the last stage must drain. During these fill and drain phases, some stages are idle, creating "pipeline bubbles."

16. **C** -- Liquid cooling improves heat removal but does not reduce leakage power, which is a property of the transistor technology and chip temperature. (In fact, by keeping chips cooler, liquid cooling can slightly reduce temperature-dependent leakage, but this is not a primary benefit.)
