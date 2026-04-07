# Training vs Inference Architectures

This section covers the architectural differences between AI accelerators optimized for training versus inference, including batch size considerations, latency vs throughput tradeoffs, KV cache management, and speculative decoding.

---

### Q1. What are the fundamental differences between training and inference workloads?

**Answer:**

Training and inference differ in ways that have profound implications for accelerator architecture:

**Batch size**: Training uses large batch sizes (thousands to millions of samples per step) to amortize weight loading and achieve high compute utilization. Inference may use batch size 1 (for latency-sensitive applications) or moderate batch sizes (for throughput-oriented serving). Large batches make training compute-bound; small batches make inference memory-bound.

**Precision**: Training requires sufficient precision for gradient computation and accumulation (typically BF16/FP16 with FP32 accumulation). Inference can use aggressively reduced precision (INT8, INT4) because the model weights are fixed and can be carefully quantized offline.

**Memory requirements**: Training stores activations for the backward pass (proportional to batch size * model size * number of layers), optimizer states (2-3x model size for Adam), and gradients. Inference stores only the model weights, the current activation, and (for autoregressive models) the KV cache.

**Compute pattern**: Training executes forward pass, backward pass, and optimizer step -- all are large GEMMs with high arithmetic intensity. Inference executes only the forward pass, and for autoregressive generation, each forward pass generates a single token with matrix-vector multiplies (low arithmetic intensity).

**Latency vs throughput**: Training optimizes for throughput (total FLOPS per second). Inference may optimize for latency (time to generate a response) or throughput (total inferences per second), depending on the application.

| Property | Training | Inference |
|---|---|---|
| Batch size | Large (1000+) | 1 to 256 |
| Precision | BF16/FP16 + FP32 | INT8/INT4/FP8 |
| Dominant bottleneck | Compute | Memory bandwidth |
| Key metric | Throughput (TFLOPS) | Latency (ms) or throughput (tok/s/$) |
| Multi-chip needs | Always | Sometimes |
| Memory per chip | 80+ GB HBM | 16-80 GB HBM |

---

### Q2. What is the KV cache and why is it a bottleneck for inference?

**Answer:**

The KV cache stores the key and value tensors from all previous tokens during autoregressive generation. In each transformer layer, the self-attention mechanism computes attention scores between the current token's query and all previous tokens' keys, then aggregates previous tokens' values weighted by these scores. Without caching, the Q, K, V projections for all previous tokens would need to be recomputed for every new token.

The KV cache size for a model with L layers, H_kv key-value heads, and head dimension D_h at sequence length S is:
```
KV cache = 2 * L * H_kv * D_h * S * bytes_per_element
```

For Llama 70B (L=80, H_kv=8 with GQA, D_h=128) at S=8192 with FP16:
```
KV cache = 2 * 80 * 8 * 128 * 8192 * 2 = 2.68 GB
```

The KV cache creates several bottlenecks:

**Memory capacity**: For long sequences or large batches of concurrent requests, the KV cache can exceed available HBM. At S=32768 with 256 concurrent requests, the aggregate KV cache would be 256 * 10.7 GB = 2.7 TB, far exceeding any single chip's HBM.

**Memory bandwidth**: Each token generation reads the entire KV cache from HBM. At S=8192, this is 2.68 GB per token per request. For 256 concurrent requests at 50 tokens/second, the KV cache bandwidth alone is 256 * 2.68 * 50 = 34.3 TB/s -- exceeding any single chip's HBM bandwidth.

**Memory management**: The KV cache grows dynamically as tokens are generated and must handle variable-length sequences. Naive memory allocation wastes space (pre-allocating for maximum sequence length) or causes fragmentation.

Solutions include: GQA/MQA (reducing H_kv), KV cache quantization (INT8 or INT4), PagedAttention (virtual memory-style KV cache management, used in vLLM), sliding window attention (capping KV cache at a window size), and distributed KV cache across multiple chips.

---

### Q3. What is speculative decoding and how does it improve inference latency?

**Answer:**

Speculative decoding is a technique that uses a small, fast "draft" model to speculatively generate multiple tokens, which are then verified in parallel by the large "target" model. If the draft model's tokens match what the target model would have generated, multiple tokens are accepted in a single forward pass, reducing latency.

The algorithm:
1. The draft model (e.g., a 1B parameter model) generates K tokens autoregressively (fast, because the draft model is small).
2. The target model (e.g., a 70B parameter model) processes all K+1 tokens (the context plus K draft tokens) in a single forward pass, computing the probability distribution at each position.
3. For each drafted token, the target model's probability is compared to the draft model's probability. Tokens are accepted with a probability that maintains the target model's exact distribution (no accuracy loss).
4. All consecutive accepted tokens (from the beginning) are kept. If the j-th token is rejected, tokens j through K are discarded, and the correct token at position j is sampled from the target model's adjusted distribution.

Why it works: The target model's forward pass for K+1 tokens takes roughly the same time as for 1 token (when batch size is small, the bottleneck is weight loading, not compute). So verifying K tokens in parallel costs approximately the same as generating 1 token normally.

Expected speedup: If the draft model's acceptance rate is p, the expected number of tokens accepted per verification step is approximately 1/(1-p). With p=0.8, the expected speedup is ~5x. With p=0.6, the speedup is ~2.5x.

Hardware implications: Speculative decoding increases compute requirements (both draft and target models run) but reduces latency because multiple tokens are generated per target model forward pass. It favors hardware with high memory bandwidth (to quickly run both models) and large batch processing capability (to efficiently process the K+1 verification tokens).

---

### Q4. How do inference-specific chips (Groq, AWS Inferentia) differ from training GPUs?

**Answer:**

Inference-specific chips make different architectural tradeoffs than training GPUs:

**Groq LPU**: Uses a deterministic, statically-scheduled architecture with 230 MB of on-chip SRAM and no HBM. The compiler pre-schedules all operations, eliminating runtime scheduling overhead and providing consistent, low latency. Optimized for batch-1 inference where predictable latency matters more than peak throughput. Does not support training (no backward pass, no floating-point accumulation for gradient computation).

**AWS Inferentia2**: Amazon's custom inference chip built on a 7nm process. Features NeuronCore-v2 engines with dedicated matrix and vector engines. 32 GB HBM2e per chip. Designed for throughput-optimized inference at lower cost per inference than GPU-based solutions. The Neuron compiler (part of AWS Neuron SDK) optimizes models for the NeuronCore architecture.

**AWS Trainium**: Amazon's training counterpart to Inferentia. Higher compute throughput and HBM bandwidth, with support for BF16/FP32 training and stochastic rounding. Includes NeuronLink for multi-chip scaling (similar to NVLink but Amazon-proprietary).

**Intel Gaudi (Habana Labs)**: Uses a heterogeneous architecture with a matrix math engine (GEMM accelerator), a tensor processing core (programmable for non-GEMM operations), and integrated RDMA NICs for scale-out. The integrated networking is a key differentiator: Gaudi 2 has 24 100GbE RDMA ports built into the chip, eliminating the need for external NICs.

Key differences from training GPUs:

| Feature | Training GPU (H100) | Inference chip (typical) |
|---|---|---|
| Peak FLOPS | Very high (990 TFLOPS FP16) | Lower (optimized for efficiency) |
| HBM capacity | Large (80 GB) | Moderate (16-32 GB) |
| Precision focus | BF16/FP16/FP8 + FP32 | INT8/INT4 focused |
| TDP | High (700W) | Lower (100-300W) |
| TOPS/W | Moderate | Higher |
| TOPS/$ | Lower | Higher for inference |
| Programming model | CUDA (flexible) | SDK-specific (less flexible) |

---

### Q5. What is continuous batching and why is it important for LLM serving?

**Answer:**

Continuous batching (also called iteration-level batching or inflight batching) dynamically adds and removes requests from the batch at every generation step, rather than waiting for all requests in a batch to complete before starting new ones.

In naive (static) batching, a batch of N requests is processed together from start to finish. If some requests finish early (short responses) or some are much longer, the shorter requests must wait until the longest one completes. This wastes compute on padding tokens and increases latency for shorter requests.

Continuous batching solves this by maintaining a dynamic batch. When a request finishes (reaches the end-of-sequence token), a new waiting request immediately takes its slot in the batch. When a new request arrives, it joins the batch at the next generation step (with its prompt tokens processed through the prefill phase first).

Benefits: (1) higher throughput -- the batch is always full, maximizing GPU utilization; (2) lower latency for short requests -- they are not held up by longer requests; (3) better resource utilization -- no wasted computation on padding.

Implementation in vLLM and TensorRT-LLM: The serving framework maintains a scheduler that manages the active batch. At each generation step, the scheduler decides which requests to include (based on their current position, available KV cache memory, and priority). PagedAttention manages the KV cache memory for variable-length sequences in the batch.

Hardware implications: Continuous batching works best on hardware that can efficiently handle variable-length sequences within a batch. This requires support for variable-length attention masking (different requests have different sequence lengths) and efficient memory management for the KV cache. GPUs handle this well due to their flexible CUDA programming model; more rigid architectures (systolic arrays with fixed schedules) may struggle with the variability.

---

### Q6. What is the prefill vs decode distinction in LLM inference?

**Answer:**

LLM inference for a single request consists of two phases with very different computational characteristics:

**Prefill phase**: The model processes the entire input prompt (all input tokens) in a single forward pass, producing the KV cache for all prompt tokens and the first output token. This is a GEMM-heavy computation with batch size equal to the prompt length (often hundreds to thousands of tokens). The arithmetic intensity is high, making the prefill phase compute-bound on modern hardware. Prefill latency is the "time to first token" (TTFT).

**Decode phase**: The model generates output tokens one at a time, each requiring a forward pass that reads the entire model weights and the growing KV cache. Each step is a matrix-vector multiply (effectively batch size 1 per request), making it extremely memory-bandwidth-bound. Decode latency per token is the "inter-token latency" (ITL) or "time per output token" (TPOT).

The computational profile differs dramatically:

| Metric | Prefill | Decode |
|---|---|---|
| Tokens processed | Prompt length (100-10000) | 1 per step |
| Arithmetic intensity | High (compute-bound) | Low (memory-bound) |
| GPU utilization | High (70-90%) | Low (5-20%) |
| Bottleneck | Compute throughput | HBM bandwidth |
| Duration per request | 10-100 ms (once) | 10-30 ms per token |

This distinction has important implications for serving system design. Prefill and decode can be disaggregated: some chips handle only prefill (high compute, short duration) while others handle only decode (high bandwidth, long duration). This "disaggregated serving" allows using different hardware configurations for each phase.

---

### Q7. How does batch size affect the compute/memory tradeoff for inference?

**Answer:**

Batch size is the most powerful knob for shifting inference between memory-bound and compute-bound regimes. The relationship is mediated through arithmetic intensity.

For a single linear layer (weight matrix D_in x D_out):
```
FLOPS = 2 * B * D_in * D_out
Weight bytes = D_in * D_out * bytes_per_weight
Input/output bytes = B * (D_in + D_out) * bytes_per_activation
Arithmetic intensity ≈ 2B / bytes_per_weight (for large D, weights dominate)
```

For FP16 weights: AI = 2B/2 = B FLOPS/byte
For INT8 weights: AI = 2B/1 = 2B FLOPS/byte

The ridge point of an H100 is approximately 295 FLOPS/byte (FP16 compute / HBM bandwidth).

| Batch size | AI (FP16 weights) | Regime on H100 |
|---|---|---|
| 1 | 1 | Memory-bound (295x below ridge) |
| 32 | 32 | Memory-bound (9.2x below) |
| 128 | 128 | Memory-bound (2.3x below) |
| 256 | 256 | Near ridge point |
| 512 | 512 | Compute-bound |

This analysis shows that batch sizes of 256+ are needed to fully utilize an H100's tensor cores for FP16 inference. At batch size 1, only 0.34% of the tensor core capability is utilized -- the remaining 99.66% sits idle waiting for data from HBM.

For throughput-optimized serving, increasing batch size is the most effective optimization. For latency-optimized serving, batch size is constrained (processing a larger batch takes longer per request), and the focus shifts to maximizing memory bandwidth through quantization and multi-chip parallelism.

---

### Q8. What are the architectural implications of Mixture of Experts (MoE) models for inference?

**Answer:**

MoE models activate only a subset of "expert" sub-networks for each token, providing conditional computation. A typical MoE layer has a router that selects the top-K (usually K=1 or K=2) experts out of E total experts (e.g., E=8 or E=64) for each token.

Architectural implications:

**Memory capacity**: MoE models have many more total parameters than dense models of equivalent quality, because most parameters are in the expert layers. Mixtral 8x7B has 47B total parameters but uses ~13B per token. All expert weights must be stored in memory even though only a fraction is used per token. This increases HBM capacity requirements.

**Memory bandwidth**: During inference, only the selected expert weights need to be loaded from HBM for each token. At batch size 1, this reduces the weight loading bandwidth compared to a dense model of the same quality. However, the benefit diminishes with larger batches (different tokens may route to different experts, and all experts may be active across the batch).

**Compute utilization**: Different experts may receive different numbers of tokens, creating load imbalance across compute units. If expert A receives 10 tokens and expert B receives 1 token, the compute units processing expert B finish early and sit idle. Load balancing techniques (auxiliary loss terms during training, token dropping, expert capacity factors) mitigate but do not eliminate this imbalance.

**Communication**: In multi-chip deployments, expert parallelism places different experts on different chips. An all-to-all communication step routes tokens to the correct expert chips, and another all-to-all returns results. This communication pattern is different from the all-reduce used in dense tensor parallelism and may require different interconnect designs.

**Hardware support**: Efficient MoE inference benefits from: fast weight switching (loading different expert weights from HBM with low overhead), dynamic routing hardware (implementing the top-K selection efficiently), and efficient all-to-all communication for expert parallelism.

---

### Q9. What is the role of model parallelism in large-model inference?

**Answer:**

When a model is too large to fit on a single chip (including weights + KV cache), model parallelism distributes the model across multiple chips. For inference, the two main approaches are:

**Tensor parallelism (TP)**: Each layer's weight matrices are split across T chips. For a linear layer with weight W of shape (D_in, D_out), each chip holds a (D_in, D_out/T) slice. Each chip computes its portion of the output, and an all-reduce (or all-gather + reduce-scatter) combines the results. TP reduces the per-chip memory and bandwidth requirement by T, but adds inter-chip communication every layer.

**Pipeline parallelism (PP)**: Different layers are assigned to different chips. The input is processed sequentially through the pipeline stages. PP reduces per-chip memory (each chip only holds its layers' weights and KV cache) but adds pipeline bubbles and inter-stage communication latency. For inference, PP is simpler than TP because the inter-stage communication is just the activation tensor passed between adjacent stages, with no all-reduce needed.

For inference, TP is preferred when inter-chip bandwidth is high (NVLink) because it reduces per-chip latency (all chips process each token in parallel). PP is preferred when bandwidth is limited (PCIe, inter-node) because it requires less communication volume.

The choice of T (number of tensor-parallel chips) is determined by: (1) the model must fit in T * per-chip-HBM, (2) the per-layer communication overhead at bandwidth B must be small relative to the compute time, and (3) T should be a power of 2 for efficient matrix partitioning.

For a 70B model on H100 GPUs (80 GB HBM each): weights require 140 GB (FP16) or 70 GB (INT8). With INT8, the model fits on 1 GPU but with little room for KV cache. With 2 GPUs (TP=2), each holds 35 GB weights, leaving 45 GB for KV cache. This is the typical deployment: TP=2 with NVLink for 70B models.

---

### Q10. What emerging hardware trends are specific to inference optimization?

**Answer:**

**Dedicated inference chips**: Companies like Groq, Furiosa, d-Matrix, and Tenstorrent are building chips specifically for inference, optimizing for TOPS/W and TOPS/$ rather than peak FLOPS. These chips typically feature large on-chip SRAM, lower TDP, and INT8/INT4-optimized datapaths.

**Processing-in-memory (PIM)**: Samsung HBM-PIM and SK Hynix AiM place compute logic within the HBM stack. For memory-bound inference workloads, this eliminates the bandwidth bottleneck by processing data where it resides. Early PIM designs support simple operations (element-wise, GEMV) within the memory.

**Chiplet-based inference**: Disaggregated designs with separate compute chiplets and memory chiplets connected via CXL or similar protocols. This allows flexibly scaling memory capacity and bandwidth independently of compute. Useful for large models with massive KV caches.

**Edge inference NPUs**: Neural Processing Units in mobile SoCs (Apple Neural Engine, Qualcomm AI Engine, Google Tensor TPU) optimize for low-power inference (1-15W). They use INT8/INT4 computation with minimal overhead, achieving 10-30 TOPS/W.

**Photonic interconnects for inference**: Companies like Lightmatter are exploring photonic computing for matrix-vector multiplication, using light-based computation that is potentially faster and more energy-efficient than electronic computation for the specific pattern of multiply-accumulate operations.

**Software-defined inference**: Rather than fixed hardware, some approaches use reconfigurable hardware (FPGAs, CGRAs) that can be adapted to different model architectures. Xilinx (now AMD) Versal AI Engines represent this approach, offering reconfigurable compute tiles optimized for different AI workload patterns.

The common theme is that inference optimization focuses on memory bandwidth efficiency, cost per inference, and energy per inference, rather than the peak FLOPS that dominate training hardware design.
