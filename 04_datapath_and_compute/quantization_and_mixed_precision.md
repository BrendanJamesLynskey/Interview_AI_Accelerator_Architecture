# Quantization and Mixed Precision

This section covers quantization techniques for AI models including post-training quantization (PTQ), quantization-aware training (QAT), mixed-precision training, and modern quantization methods like GPTQ and AWQ.

---

### Q1. What is post-training quantization (PTQ) and how does it work?

**Answer:**

Post-training quantization converts a pre-trained floating-point model to lower precision (typically INT8 or INT4) without any retraining. The process involves: (1) running a small calibration dataset through the model to collect statistics (min, max, or histogram) of activations at each layer, (2) computing optimal quantization parameters (scale and zero-point) for each tensor based on these statistics, and (3) converting the model weights and optionally activations to the target integer format.

For weight quantization, the process is straightforward because weights are fixed after training. The scale factor is computed as: `scale = (max_val - min_val) / (2^bits - 1)`. For INT8 with symmetric quantization: `scale = max(abs(weights)) / 127`.

For activation quantization, the challenge is that activation ranges are input-dependent. The calibration dataset is used to estimate the typical activation range. Methods include MinMax calibration (using the observed min/max values), Entropy calibration (choosing a range that minimizes KL divergence between the float and quantized distributions), and Percentile calibration (clipping the top/bottom p% of values to reduce the impact of outliers).

PTQ typically achieves less than 1% accuracy degradation for INT8 on most models (CNNs, standard transformers). For INT4, accuracy loss is more significant (1-5%), especially for large language models where certain channels have outlier values that are poorly represented in low bit widths.

PTQ is attractive because it requires no training compute -- only a small calibration dataset and a few minutes of processing. This makes it the default approach for deploying existing models on inference hardware.

---

### Q2. What is quantization-aware training (QAT) and when is it necessary?

**Answer:**

Quantization-aware training simulates the effects of quantization during the training process, allowing the model to adapt its weights to be robust to quantization noise. This typically achieves better accuracy than PTQ, especially at very low bit widths (4-bit, 2-bit).

During QAT, quantize-dequantize (fake quantization) operations are inserted into the computation graph at the points where quantization will occur during inference. In the forward pass, tensors pass through these fake quantization nodes that round values to the nearest quantized level and then dequantize back to float. This introduces quantization noise that the gradient descent process learns to compensate for.

The challenge is that the rounding operation in quantization is not differentiable (its gradient is zero almost everywhere). QAT uses the Straight-Through Estimator (STE), which approximates the gradient of the rounding function as 1 (passing the gradient through unchanged). Despite being theoretically questionable, STE works well in practice because the model's loss landscape is smooth enough that approximate gradients still drive useful optimization.

QAT is necessary when: (1) PTQ produces unacceptable accuracy loss (common at INT4 and below), (2) the model has layers with unusual activation distributions that calibration-based PTQ handles poorly, or (3) the deployment requires the absolute best accuracy at a given quantization level.

The cost of QAT is that it requires access to the training data and compute resources for fine-tuning (typically 10-20% of the original training cost). For large language models with billions of parameters, this cost can be significant, which has motivated the development of efficient PTQ methods like GPTQ and AWQ.

---

### Q3. What is GPTQ and how does it achieve high-quality weight quantization?

**Answer:**

GPTQ (GPT Quantization) is a one-shot weight quantization method that achieves near-QAT accuracy with PTQ-level compute cost. It was introduced in 2022 and has become one of the most widely used methods for quantizing large language models to 4-bit and 3-bit weights.

GPTQ is based on the Optimal Brain Quantization (OBQ) framework, which quantizes weights one at a time while adjusting the remaining unquantized weights to compensate for the quantization error. The key insight is that the quantization error of one weight can be partially absorbed by adjusting the other weights in the same row using the Hessian (second-order) information of the layer's loss function.

The algorithm processes each row of the weight matrix independently:
1. Compute the Hessian inverse H^{-1} for the layer using a small calibration dataset (the Hessian of the squared error with respect to the weights).
2. For each column (weight) in order:
   a. Quantize the weight to the nearest quantized level.
   b. Compute the quantization error.
   c. Update all remaining unquantized weights in the row using: `delta_w = -error * H^{-1}[remaining, col] / H^{-1}[col, col]`.
3. This update distributes the quantization error across the remaining weights, minimizing the total layer output error.

GPTQ can quantize a 175B parameter model to 4-bit in approximately 4 GPU-hours (vs weeks for QAT), with accuracy degradation of less than 1% on most benchmarks. It is particularly effective for weight-only quantization (keeping activations in FP16), which is the most common inference deployment scenario.

The method's efficiency comes from processing entire rows in parallel and using Cholesky decomposition of the Hessian for numerical stability. GPTQ has been integrated into popular libraries (AutoGPTQ, llama.cpp, Hugging Face Transformers) for easy deployment.

---

### Q4. What is AWQ (Activation-aware Weight Quantization)?

**Answer:**

AWQ is a weight quantization method that identifies and protects the most important weight channels based on activation magnitudes. It was introduced in 2023 and achieves competitive accuracy with GPTQ while being simpler and faster.

The key observation is that not all weight channels are equally important for model accuracy. AWQ measures importance by the magnitude of the corresponding activations: a weight channel that multiplies large activations has a larger impact on the output than one that multiplies small activations. Quantizing important channels with the same uniform scheme as unimportant channels disproportionately degrades accuracy.

AWQ addresses this by per-channel scaling before quantization:
1. Analyze a calibration dataset to compute the average activation magnitude for each input channel.
2. Scale up the weights in important channels (those with large activation magnitudes) and scale down the corresponding input activations, so that the mathematical result is unchanged but the important weights occupy a larger fraction of the quantization range.
3. Apply standard uniform quantization to the scaled weights.

The scaling effectively allocates more quantization precision to important channels. Since the scaling is absorbed into the weight matrix (and the inverse scaling into the preceding layer's output or the activation), there is no runtime overhead.

AWQ advantages over GPTQ: (1) no Hessian computation needed (only activation statistics), (2) faster processing time, (3) compatible with any quantization backend (GPTQ requires specialized kernels for its non-uniform weight update pattern). AWQ achieves similar accuracy to GPTQ at 4-bit quantization and is widely used in deployment (available in llama.cpp, vLLM, and TensorRT-LLM).

---

### Q5. How does mixed-precision training achieve near-FP32 accuracy with half the memory?

**Answer:**

Mixed-precision training, formalized by Micikevicius et al. (NVIDIA, 2018), uses FP16 or BF16 for the majority of computation and storage while maintaining FP32 for critical operations. The approach rests on three pillars:

**FP32 master weights**: A master copy of all weights is maintained in FP32. At the start of each forward pass, weights are cast to FP16/BF16. After the backward pass, the FP16/BF16 gradients are used to update the FP32 master weights. This ensures that small gradient updates (which might be rounded to zero in FP16) are accumulated accurately over many iterations.

**FP32 accumulation**: Matrix multiplications use FP16/BF16 inputs but accumulate partial sums in FP32. Modern tensor cores do this natively: the multiplier takes FP16/BF16 operands but the accumulation register is FP32. This prevents loss of precision when summing many small products.

**Loss scaling (for FP16 only)**: FP16's limited dynamic range means that gradient values smaller than ~6*10^-8 underflow to zero. Since many gradient values fall in this range (especially in early layers), loss scaling multiplies the loss by a large factor before backpropagation, shifting all gradients into the representable range. After gradient computation, the scale is removed before the weight update.

Memory savings: Activations (the largest memory consumer during training) are stored in FP16/BF16, halving their size. Weight storage increases by 50% (FP16 copy + FP32 master copy vs FP32 only), but activations typically dominate memory. Optimizer states remain in FP32. Net memory savings are approximately 30-40% compared to pure FP32 training.

Performance improvement: Tensor core operations in FP16/BF16 deliver 2x the throughput of TF32 and 16x the throughput of FP32 (non-tensor) on modern NVIDIA GPUs. Combined with reduced memory bandwidth requirements (half the bytes per activation), mixed-precision training typically provides 1.5-3x speedup over pure FP32 training with negligible accuracy difference.

---

### Q6. What is weight-only quantization and why is it popular for LLM inference?

**Answer:**

Weight-only quantization keeps the weights in low precision (INT4, INT8) but performs the computation in higher precision (FP16). During inference, the quantized weights are dequantized on-the-fly to FP16 before the matrix multiplication.

This approach is popular for LLM inference because of the specific bottleneck structure of autoregressive generation. At batch size 1, each fully connected layer is a matrix-vector multiply where the entire weight matrix must be loaded from HBM for a single input vector. The bottleneck is entirely memory bandwidth (loading the weights), not compute (the actual matrix-vector multiply uses a tiny fraction of available FLOPS).

Quantizing weights from FP16 to INT4 reduces the weight data by 4x, directly translating to 4x less HBM bandwidth and approximately 4x faster inference for bandwidth-bound operations. The dequantization step (converting INT4 to FP16 in the compute pipeline) adds minimal overhead because it is performed in registers/shared memory and is compute-light compared to the matrix multiplication itself.

The accuracy-efficiency tradeoff for weight-only quantization is favorable because: (1) weights have a fixed, known distribution that can be carefully calibrated, (2) the dequantization maintains FP16 arithmetic for the actual computation, avoiding the accumulation precision issues of full INT4 computation, and (3) modern quantization methods (GPTQ, AWQ) achieve excellent accuracy at 4-bit weights.

Weight-only quantization has become the standard approach for deploying large language models on consumer hardware. A 70B parameter model that would require 140 GB in FP16 (needing 2+ A100 GPUs) can be quantized to INT4 at 35 GB, fitting on a single GPU with room for the KV cache.

---

### Q7. What are group quantization and sub-channel quantization?

**Answer:**

Group quantization divides a weight tensor into small groups of consecutive elements, each with its own scale factor (and optionally zero-point). This provides finer-grained quantization than per-tensor or per-channel approaches, reducing quantization error at the cost of storing additional scale factors.

For a weight matrix of shape (M, K) with group size G:
- Per-tensor: 1 scale factor total
- Per-channel: M scale factors (one per output channel, applied to each row)
- Group quantization with G=128: M * (K/128) scale factors

Group quantization is critical for INT4 and lower-bit quantization of LLMs. At 4-bit precision with per-tensor quantization, outlier weights (which can be 10-100x larger than typical weights) force the scale factor to cover a wide range, wasting most of the 16 representable values on the small-magnitude majority. With group size 128, each group of 128 weights has its own scale that tightly covers its local range.

The storage overhead for group quantization is modest. For INT4 with group size 128 and FP16 scales:
```
Weight bits per element: 4 bits + 16 bits / 128 = 4.125 bits
Overhead: 3.1% compared to pure 4-bit
```

Sub-channel quantization is a more general term for any quantization granularity finer than per-channel. It includes group quantization as well as per-block (2D groups) and per-tile approaches. The optimal granularity depends on the weight value distribution and the hardware's ability to efficiently handle per-group dequantization.

Hardware support for group quantization requires the dequantization logic to look up the correct scale factor for each group before multiplying. This is straightforward in software (an index lookup per group) but requires specific hardware support for full throughput. NVIDIA's TensorRT and recent GPU architectures include optimized kernels for group-quantized matrix multiplication.

---

### Q8. What is the accuracy impact of quantization on different model types?

**Answer:**

The impact of quantization varies significantly across model types, sizes, and target precisions:

**CNNs (image classification)**: Highly tolerant of quantization. INT8 PTQ typically loses less than 0.5% top-1 accuracy on ImageNet. Even INT4 QAT can maintain within 1-2% of FP32 accuracy. CNNs' tolerance stems from their heavy use of batch normalization (which absorbs scale changes) and the redundancy in overparameterized architectures.

**Small transformers (BERT, DistilBERT)**: Moderately tolerant. INT8 PTQ maintains accuracy well. INT4 requires QAT or advanced PTQ methods for acceptable accuracy.

**Large language models (7B-70B+)**: The relationship between model size and quantization tolerance is nuanced. Larger models tend to tolerate quantization better in relative terms (percentage accuracy loss) because they have more redundancy. However, absolute sensitivity to outlier channels increases with model size. INT8 PTQ works well for most LLMs. INT4 with group quantization (GPTQ/AWQ) maintains good accuracy on most benchmarks with group sizes of 128.

**Specific challenges for LLMs**:
- **Outlier channels**: Some transformer layers develop channels with activation magnitudes 10-100x larger than average. These outliers cause disproportionate quantization error. Solutions include per-channel quantization, mixed-precision (keeping outlier channels in FP16), and SmoothQuant (mathematically migrating the quantization difficulty from activations to weights).
- **Emergent abilities**: Very low-bit quantization (3-bit, 2-bit) can disproportionately degrade LLMs' emergent abilities (multi-step reasoning, code generation) more than simple benchmarks suggest. Perplexity might change little, but downstream task performance can drop significantly.

**Diffusion models**: Moderately tolerant. INT8 quantization of the U-Net denoiser works well. The challenge is quantizing across the different noise levels (timesteps), as the activation distribution changes significantly.

---

### Q9. What is SmoothQuant and how does it address the activation outlier problem?

**Answer:**

SmoothQuant is a technique that mathematically migrates the quantization difficulty from activations to weights, enabling efficient INT8 quantization of both weights and activations in transformer models. It was introduced by Xiao et al. in 2022.

The problem: In large transformers, certain channels in the activations develop outlier values that are 10-100x larger than typical values. These outliers make activation quantization challenging because the quantization scale must cover the outlier range, reducing the effective precision for the majority of non-outlier values.

The insight: Weights do not have outliers (their distribution is smooth and well-behaved). If the quantization difficulty could be transferred from activations to weights, both could be quantized effectively.

The method: For a linear layer Y = X * W, SmoothQuant introduces a per-channel smoothing factor s:
```
Y = X * W = (X * diag(s)^{-1}) * (diag(s) * W)
```

The smoothing factor s is chosen to equalize the difficulty of quantizing the smoothed activations X' = X * diag(s)^{-1} and the smoothed weights W' = diag(s) * W. If channel j has large activation outliers, s[j] is set large, which divides the activations by s[j] (reducing outliers) and multiplies the weights by s[j] (increasing their magnitude). The mathematical result is unchanged.

The optimal s[j] is: `s[j] = max(|X[:, j]|)^alpha / max(|W[j, :]|)^(1-alpha)`, where alpha (typically 0.5) controls the migration balance. When alpha = 0.5, the quantization difficulty is equally shared.

SmoothQuant enables W8A8 (8-bit weights and 8-bit activations) quantization of LLMs like OPT-175B with less than 1% accuracy loss. The key advantage over weight-only quantization is that with both weights and activations in INT8, the entire matrix multiplication can be performed on INT8 tensor cores, achieving 2x the throughput of FP16 tensor cores (vs weight-only quantization which still uses FP16 compute).

---

### Q10. What hardware support is needed for efficient low-bit (INT4, INT2) inference?

**Answer:**

Efficient sub-8-bit inference requires specific hardware features beyond standard INT8 MAC arrays:

**Sub-byte data handling**: INT4 values pack two elements per byte. The hardware must efficiently extract individual 4-bit values from packed storage. This requires nibble-extraction logic before the multiplier and careful address generation for packed memory access.

**Wider multiplier reuse**: A single INT8 multiplier can be repurposed for two INT4 multiplies using subword parallelism. The 8-bit multiplier is split into two 4-bit multipliers operating in parallel on the upper and lower nibbles. This requires careful carry propagation management to prevent cross-contamination between the two independent products.

**Group dequantization**: With group quantization, each group of (e.g., 128) weights shares a scale factor stored in FP16. The hardware must: (a) read the packed INT4 weights, (b) look up the corresponding group's FP16 scale, (c) dequantize by multiplying the INT4 value by the scale, and (d) perform the FP16 multiply-accumulate. This lookup and multiply must be pipelined to avoid bottlenecking the MAC throughput.

**Mixed-precision MAC**: Some designs use INT4 multiplication with FP16 or FP32 accumulation. The INT4 * INT4 product is an 8-bit integer that must be converted to FP16/FP32 for accumulation. This conversion adds a small amount of logic.

**Memory system alignment**: INT4 data is half the density of INT8, so the memory system must handle half-byte granularity. Cache lines, memory bus widths, and DMA transfer sizes must be designed to efficiently move and store sub-byte data.

**Sparse-quantization co-design**: Combining 4-bit quantization with 2:4 structured sparsity yields effective 2-bit representation (4-bit values with 50% zeros). Hardware that supports both features simultaneously can achieve 4x compression and throughput improvement over INT8 dense computation.

Current hardware support: NVIDIA Hopper and Blackwell support INT4 tensor core operations. Qualcomm's Hexagon DSP supports INT4. Many edge AI accelerators (NPUs in mobile SoCs) support INT4 and even INT2 for maximum efficiency.
