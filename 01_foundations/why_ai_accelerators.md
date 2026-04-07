# Why AI Accelerators

This section covers the fundamental motivations behind domain-specific hardware accelerators for artificial intelligence workloads, including the end of Dennard scaling, the slowdown of Moore's law, and the rise of domain-specific architectures.

---

### Q1. What is Dennard scaling and why did its end matter for AI hardware?

**Answer:**

Dennard scaling, articulated by Robert Dennard in 1974, observed that as transistors shrink, their power density remains roughly constant because voltage and current scale down proportionally with transistor dimensions. This meant that each new process node could deliver both more transistors and higher clock frequencies without increasing power consumption per unit area.

Dennard scaling effectively ended around 2006 because transistor threshold voltages could not continue to scale without causing unacceptable levels of leakage current. As gate oxide thicknesses approached a few atomic layers, quantum tunneling effects made it impossible to reduce voltage further while maintaining reliable switching behavior.

The consequence was the so-called "power wall." Clock frequencies plateaued around 3-5 GHz for general-purpose CPUs. Architects could no longer rely on voltage scaling to deliver performance improvements for free. This forced the industry toward multi-core designs and, eventually, toward domain-specific accelerators that achieve performance gains through architectural specialization rather than frequency scaling. For AI workloads, which exhibit massive data parallelism and regular computation patterns, this shift created the opening for GPUs, TPUs, and custom ASICs that trade general-purpose flexibility for orders-of-magnitude improvements in performance per watt on targeted workloads.

---

### Q2. How has Moore's law evolved, and what does its slowdown mean for accelerator design?

**Answer:**

Moore's law, Gordon Moore's 1965 observation that the number of transistors on an integrated circuit doubles approximately every two years, held remarkably well for several decades. However, beginning around the 14nm node (circa 2014), the cadence of doubling slowed significantly. The transition from 7nm to 5nm to 3nm has taken longer and cost substantially more per transistor than historical trends predicted.

The slowdown has several implications for AI accelerator design. First, architects can no longer rely on process technology alone to deliver generational performance improvements. This elevates the importance of microarchitectural innovation: better dataflow, more efficient memory hierarchies, and smarter scheduling of computation. Second, the cost per transistor has stopped falling at advanced nodes (below 7nm), which means that simply making chips bigger or denser is not economically viable without corresponding improvements in utilization and efficiency. Third, packaging innovations such as chiplets, 2.5D interposers, and 3D stacking (as seen in HBM) have become critical for scaling performance beyond what a single monolithic die can deliver.

For AI accelerators specifically, the Moore's law slowdown has motivated designs that extract maximum work per transistor through specialization. A general-purpose CPU transistor budget includes branch predictors, out-of-order execution logic, and speculative execution hardware that is irrelevant for matrix multiplication. An AI accelerator reclaims that silicon area for multiply-accumulate arrays, on-chip SRAM, and data movement logic, yielding 10-1000x better performance per watt on targeted workloads.

---

### Q3. What is the key insight behind domain-specific architectures (DSAs)?

**Answer:**

The central insight is that general-purpose processors pay a massive overhead in energy and area for flexibility that most workloads do not need. A general-purpose CPU spends roughly 5-10% of its energy on actual computation and the remaining 90-95% on instruction fetch, decode, branch prediction, register renaming, out-of-order scheduling, and data movement. For a workload like matrix multiplication, where the computation pattern is perfectly regular and fully known at compile time, all of that control overhead is wasted.

Domain-specific architectures eliminate this waste by hardwiring the control logic for a narrow class of computations. A systolic array, for example, has a fixed data movement pattern -- operands flow through a grid of processing elements in a predetermined schedule. There is no branch predictor because there are no branches. There is no instruction decoder because every processing element performs the same multiply-accumulate operation every cycle. The energy per operation drops by orders of magnitude compared to a CPU.

The tradeoff is reduced flexibility. A TPU's systolic array cannot efficiently execute a sorting algorithm or a graph traversal. The bet that DSA designers make is that the target workload (deep learning, in this case) is large enough and important enough to justify dedicating silicon to it. Given that AI training and inference now consume a significant and growing fraction of global data center compute, this bet has clearly paid off.

---

### Q4. What is the "Jevons paradox" as it applies to AI compute demand?

**Answer:**

The Jevons paradox, originally observed in the context of coal consumption in 19th-century England, states that improvements in the efficiency of resource use tend to increase rather than decrease total resource consumption, because lower costs stimulate greater demand. In the context of AI compute, this paradox plays out dramatically.

Each generation of AI accelerator delivers better performance per watt and lower cost per operation. One might expect this to reduce total energy consumption for AI workloads. Instead, the reduced cost of compute enables the training of ever-larger models (GPT-4 required an estimated 10,000-25,000 A100-equivalent GPU-months), the deployment of AI inference at scale in products used by billions of people, and the exploration of new applications that were previously computationally infeasible.

The result is that total AI compute demand has been growing at approximately 4-5x per year, far outstripping the roughly 2x per generation improvements in accelerator efficiency. This relentless demand growth is one of the primary economic drivers behind the continued investment in AI accelerator architecture. It means that even significant improvements in efficiency do not saturate the market -- they expand it. Companies like NVIDIA, Google, AMD, and a host of startups continue to find a growing market for each successive generation of AI hardware.

---

### Q5. How do AI accelerators compare to CPUs and FPGAs for deep learning workloads?

**Answer:**

CPUs are the most flexible option but the least efficient for deep learning. A modern server CPU might deliver 1-5 TFLOPS of FP32 compute at 200-350W, yielding roughly 5-15 GFLOPS/W. The vast majority of the chip's transistor budget and power is consumed by features irrelevant to matrix math: deep out-of-order pipelines, large multi-level caches optimized for pointer-chasing workloads, branch predictors, and complex coherence protocols.

FPGAs occupy a middle ground. They offer reconfigurable logic that can be tailored to specific computation patterns, achieving better efficiency than CPUs for regular workloads. A high-end FPGA can deliver 50-200 TOPS of INT8 compute. FPGAs excel in low-latency inference scenarios and in prototyping new accelerator architectures. However, they suffer from lower clock frequencies (typically 200-500 MHz vs 1-2 GHz for ASICs), lower compute density (because configurable routing consumes significant area), and a challenging programming model (RTL or high-level synthesis).

Fixed-function AI accelerators (GPUs with tensor cores, TPUs, custom ASICs) offer the best efficiency. An NVIDIA H100 delivers approximately 990 TFLOPS of FP16 tensor core compute at 700W, yielding roughly 1400 GFLOPS/W -- a 100x improvement over a CPU. Google's TPU v5e targets inference efficiency with even better TOPS/W figures. Custom ASICs from companies like Groq and Cerebras push efficiency further for specific deployment scenarios.

The choice between these platforms involves tradeoffs among flexibility, efficiency, time-to-market, and total cost of ownership. For training large models, GPU clusters dominate. For high-volume inference, custom ASICs are increasingly attractive.

---

### Q6. What role did the ImageNet moment (2012) play in driving AI accelerator development?

**Answer:**

In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton demonstrated that a deep convolutional neural network (AlexNet) trained on GPUs could dramatically outperform all prior approaches on the ImageNet image classification benchmark, reducing the top-5 error rate from 26% to 16%. This result was pivotal for several reasons.

First, it demonstrated that GPUs, originally designed for graphics rendering, were remarkably well-suited to the parallel matrix operations at the heart of neural network training. AlexNet was trained on two NVIDIA GTX 580 GPUs, each with 512 CUDA cores. The key insight was that the same SIMD-style parallelism that enables real-time rendering of millions of pixels also enables efficient computation of millions of neuron activations.

Second, the result triggered an explosion of research in deep learning, which in turn generated enormous demand for GPU compute. NVIDIA recognized this opportunity and pivoted a significant portion of its R&D toward deep learning acceleration, leading to the introduction of cuDNN (2014), tensor cores (Volta, 2017), and the modern data center GPU lineup (A100, H100, B200).

Third, and most importantly for accelerator architecture, the ImageNet moment established a clear, measurable relationship between compute investment and model quality. Researchers discovered scaling laws suggesting that model accuracy improves predictably with more compute, more data, and more parameters. This created a seemingly unlimited appetite for compute that has driven the AI accelerator industry ever since.

---

### Q7. What are the key metrics used to evaluate AI accelerators?

**Answer:**

The primary metrics fall into several categories:

**Raw compute throughput** is measured in FLOPS (floating-point operations per second) or TOPS (tera operations per second, typically for integer). Peak throughput indicates the theoretical maximum if every compute unit is busy every cycle. Common figures include FP32, FP16, BF16, FP8, and INT8 throughput, since different precisions yield different peak numbers.

**Energy efficiency** is measured in FLOPS/W or TOPS/W. This is arguably the most important metric for both data center (where power and cooling are dominant costs) and edge (where battery life matters) deployments. A useful related metric is total cost of ownership (TCO) per unit of useful compute.

**Memory bandwidth** is measured in GB/s or TB/s. For memory-bound operations (which many inference workloads are), memory bandwidth rather than compute throughput determines actual performance. The ratio of compute throughput to memory bandwidth (in ops/byte) defines the hardware's "ridge point" on the roofline model.

**Utilization** measures how much of the peak throughput is actually achieved on real workloads. A chip with 1000 TFLOPS peak but 10% utilization on a target workload delivers only 100 TFLOPS of useful compute. Utilization depends on the memory hierarchy, data movement efficiency, and software stack quality.

**Latency** matters for inference, especially in interactive applications. It includes compute latency, memory access latency, and communication latency for distributed workloads.

**Scalability** describes how performance scales when multiple accelerators are connected. This depends on interconnect bandwidth, communication overhead, and the efficiency of collective operations.

---

### Q8. Why is the ratio of compute growth to memory bandwidth growth a fundamental challenge?

**Answer:**

Compute throughput on AI accelerators has been growing at approximately 2-3x per generation (every 2 years), while memory bandwidth has been growing at only 1.3-1.5x per generation. This widening gap, often called the "memory wall" or "bandwidth wall," means that an increasing fraction of workloads become memory-bound rather than compute-bound over time.

To illustrate: the NVIDIA A100 (2020) offered 312 TFLOPS of FP16 tensor core compute with 2 TB/s of HBM2e bandwidth, giving a ratio of 156 ops/byte. The H100 (2022) offered 990 TFLOPS of FP16 with 3.35 TB/s of HBM3, giving a ratio of 295 ops/byte. Any operation with arithmetic intensity below 295 ops/byte is bandwidth-bound on the H100, meaning the compute units sit idle waiting for data.

Many important operations in modern neural networks -- attention mechanisms, element-wise operations, normalization layers, embedding lookups -- have low arithmetic intensity and are therefore bandwidth-bound. As models evolve toward architectures with more attention and less pure convolution, this problem worsens.

Accelerator architects attack this gap through multiple strategies: larger on-chip SRAM (to reduce off-chip accesses), data reuse techniques (tiling, double buffering), computation reordering (operator fusion to keep data on-chip between operations), reduced precision (which increases effective ops/byte), and sparsity exploitation (which reduces the number of bytes that need to be fetched). Understanding and reasoning about the compute-to-bandwidth ratio is essential for AI accelerator interviews.

---

### Q9. What is the significance of the transformer architecture for accelerator design?

**Answer:**

The transformer architecture, introduced in the 2017 "Attention Is All You Need" paper, has become the dominant model architecture for natural language processing, and increasingly for vision, audio, and multimodal tasks. Its characteristics have profound implications for accelerator design.

First, transformers are dominated by matrix multiplications (GEMMs). The self-attention mechanism involves computing Q*K^T (query-key dot products), applying softmax, and then multiplying by V (values). Each of these steps is a large GEMM. The feed-forward network (FFN) layers are also GEMMs. This GEMM-heavy workload is well-suited to systolic arrays and tensor cores.

Second, the self-attention mechanism has quadratic complexity in sequence length (O(n^2) for both compute and memory), creating unique challenges for long-context models. This has driven research into memory-efficient attention implementations (FlashAttention), hardware support for fused attention kernels, and architectural innovations like sparse attention patterns.

Third, the KV cache in autoregressive inference creates a memory capacity bottleneck. For a large language model generating tokens one at a time, the KV cache grows linearly with sequence length and must be stored in fast memory. This shifts the bottleneck from compute to memory capacity and bandwidth, motivating designs with large HBM capacity and high bandwidth.

Fourth, the scale of transformer models (hundreds of billions to trillions of parameters) necessitates distributed training and inference across multiple accelerators, making interconnect bandwidth and collective communication efficiency first-class design concerns.

---

### Q10. What is meant by "hardware lottery" in the context of AI accelerators?

**Answer:**

The "hardware lottery," a term coined by Sara Hooker in her 2020 paper, refers to the phenomenon where certain research ideas succeed or fail not because of their intrinsic merit but because of whether they align well with the available hardware. An algorithm that maps efficiently onto existing accelerators will be faster to train, easier to scale, and therefore more likely to be adopted and refined by the research community, regardless of whether alternative algorithms might be theoretically superior.

For example, dense matrix multiplication maps extremely well onto GPU tensor cores and TPU systolic arrays. As a result, architectures built on dense GEMMs (transformers, MLPs) have flourished, while architectures that rely on sparse or irregular computation patterns (graph neural networks, certain forms of dynamic routing) have received comparatively less attention, partly because they are harder to accelerate on current hardware.

The hardware lottery creates a feedback loop: researchers gravitate toward hardware-friendly algorithms, hardware designers optimize for the algorithms researchers use, and alternative approaches are disadvantaged. This has important implications for accelerator architects, who must decide whether to optimize for today's dominant workloads (potentially reinforcing the lottery) or to build more flexible hardware that enables a broader range of algorithmic innovation.

Understanding the hardware lottery is valuable in interviews because it demonstrates awareness of the co-evolution of algorithms and hardware, and it informs design decisions about how much flexibility to build into an accelerator versus how much to specialize for current workloads.
