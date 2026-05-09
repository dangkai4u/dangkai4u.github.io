---
title: "Towards Low-Precision Computation in LLMs: From BF16 to FP4"
date: 2026-05-01
lastmod: 2026-05-01
draft: false
keywords: ["LLM", "low-precision", "FP8", "FP4", "quantization", "mixed precision"]
description: "A practical walkthrough of low-precision numerical formats, scaling strategies, and precision loss mitigation in LLMs, grounded in recent work from DeepSeek-V3/V4 and GPT-OSS."
tags: ["LLM", "Low-Precision Computing", "Quantization"]
categories: ["AI"]
author: "dangkai.dk"
toc: true
comment: false
autoCollapseToc: false
postMetaInFooter: true
hiddenFromHomePage: false
contentCopyright: false
reward: false
mathjax: true
---

In the trillion-parameter era, low-precision computation is no longer a compromise on accuracy — it is a deliberate engineering choice.

<!--more-->

## 1. Low Precision Is Becoming the Default

Scaling laws continue to hold: model sizes have grown from tens of billions to hundreds of billions to trillions of parameters, and the compute and memory costs for training and inference have surged accordingly. Yet GPU memory and compute capacity cannot keep pace with the rate of model growth. Using low precision to store parameters or perform computation has become the industry's inevitable response. From BF16 at 2 bytes to FP8 at 1 byte to FP4 at 0.5 byte, each step down nearly halves both memory footprint and compute cost.

The impact is tangible. DeepSeek-V3's 671B parameters require over 1.3TB of memory in BF16, necessitating at least two-node deployment; switching to FP8 compresses this to roughly 685GB, fitting on a single 8×H200 machine [1]. GPT-OSS's 120B parameters, with MXFP4-quantized MoE weights, fit entirely on a single 80GB GPU [3]. And DeepSeek-V4-Pro, at 1.6T parameters with MoE Experts stored in FP4 and other modules in FP8, can run inference on a single 8×B200 node [2].

These are not just inference-time compression tricks. DeepSeek-V3's FP8 runs throughout the entire training process, with loss closely tracking the BF16 baseline [1]. DeepSeek-V4 embeds FP4 into the training pipeline itself via Quantization-Aware Training, rather than applying quantization post hoc [2]. Low precision is evolving from an inference optimization into part of the training paradigm.

How do we control the quality loss? The following covers three dimensions: numerical formats, scaling, and precision loss mitigation.

---

## 2. Numerical Formats and Representational Capacity

### 2.1 What Each Format Can Represent

A floating-point number consists of three components: sign bit (S), exponent bits (E), and mantissa bits (M). The exponent determines how large or small a number can be (dynamic range), while the mantissa determines the spacing between adjacent representable values (precision resolution).

| Format | Bits | S+E+M | Representable Values | Max Value | Per-Parameter Cost |
|--------|------|-------|---------------------|-----------|-------------------|
| FP32 | 32 | 1+8+23 | ~4.3 billion | 3.4×10³⁸ | 4 bytes |
| BF16 | 16 | 1+8+7 | 65,536 | 3.4×10³⁸ | 2 bytes |
| FP16 | 16 | 1+5+10 | 65,536 | 65,504 | 2 bytes |
| FP8 E4M3 | 8 | 1+4+3 | 256 | 448 | 1 byte |
| FP8 E5M2 | 8 | 1+5+2 | 256 | 57,344 | 1 byte |
| FP4 E2M1 | 4 | 1+2+1 | 16 | 6 | 0.5 byte |

Early deep learning used FP32 — 4 bytes per parameter, more than enough precision but enormous compute and storage overhead. Mixed Precision Training, proposed in 2018 [5], uses FP16 for forward and backward passes to achieve a 2× throughput boost, while maintaining FP32 master weights for parameter updates, balancing speed and accuracy. This quickly became standard practice.

However, FP16 revealed clear limitations for large model training. With only 5 exponent bits, FP16's dynamic range caps at 65,504. As models scale up, gradients frequently produce extreme values that exceed this range — overflow triggers loss spikes, underflow zeros out gradients. **BF16** was designed precisely for this: it retains the same 8-bit exponent as FP32 (dynamic range up to 3.4×10³⁸), at the cost of reducing the mantissa from 23 to 7 bits. For training, dynamic range matters far more than mantissa precision — gradients that neither overflow nor vanish allow training to proceed stably. BF16 thus rapidly replaced FP16 as the default precision for large model training.

**FP8** compresses further to 8 bits. Taking E4M3 as an example, it has only 256 representable values spread across [-448, 448]. Model weights typically cluster around ±0.01 — without scaling, only a handful of FP8 values fall in this narrow range, and many distinct weights get rounded to the same value. FP8 must be paired with scaling, as we'll detail below.

**FP4** is even more extreme: only 16 values total, with the positive representable values being {0, 0.5, 1, 1.5, 2, 3, 4, 6}. With such limited representational capacity, multi-level scaling and more sophisticated quantization strategies are essential.

It is worth noting that FP8 has two sub-formats with different roles: **E4M3** (4 exponent + 3 mantissa bits) offers higher precision and is typically used for weight and activation storage in the forward pass; **E5M2** (5 exponent + 2 mantissa bits) provides a wider dynamic range (max 57,344), better suited for gradients whose value range fluctuates more than weights. Notably, DeepSeek-V3 chose to use E4M3 exclusively, relying on fine-grained scaling to compensate for the reduced dynamic range — a fairly aggressive choice [1].

### 2.2 FP vs INT: Two Quantization Paths

Two technical routes exist in quantization: floating-point (FP) and integer (INT). Their core difference lies in how values are distributed:

```
FP8's 256 values (schematic, positive half only):
|||||||| | | |  |  |   |    |     |          |
0     dense region            sparse region
      (small spacing near 0)  (large spacing far from 0)

INT8's 256 values (schematic):
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
-128        -64         0          64        127
            uniform spacing everywhere
```

Neural network weights and activations typically follow a bell-shaped distribution — most values cluster near zero, with a few outliers at the tails. FP's logarithmic spacing naturally provides higher resolution near zero, matching this distribution. INT distributes representational capacity uniformly across the entire range, wasting capacity in sparse tail regions while offering less resolution near zero than FP formats.

So what is INT's value? **Hardware cost.** An FP8 FMA (fused multiply-add) unit occupies 40-50% more die area than INT8, with higher power consumption. On edge devices, mobile, and dedicated inference chips, INT remains the mainstream choice. Additionally, INT toolchains are more mature (GPTQ, AWQ, bitsandbytes), and virtually all GPUs support INT8 computation.

**In short: choose FP for precision-sensitive scenarios, INT when hardware-constrained.**

---

## 3. Scaling: Compensating for Limited Dynamic Range

### 3.1 Why Scaling Is Needed

Low-precision formats have limited representable values. To make these limited values faithfully reproduce the high-precision value distribution, scaling maps data into the format's effective representation range. Taking FP8 E4M3 as an example:

```
BF16 representable range: [-3.4×10³⁸, 3.4×10³⁸], 65536 values
FP8 E4M3 representable range: [-448, 448], 256 values

Problem: BF16 has 8 exponent + 7 mantissa bits; FP8 E4M3 has only 4 + 3.
Directly truncating BF16 to FP8 loses mantissa precision (7→3 bits),
and the exponent reduction (8→4 bits) may cause overflow or underflow.
```

Scaling maps the original value distribution into the low-precision format's effective representation space, making full use of its limited capacity. The implementation is straightforward: compute the maximum absolute value `amax` of a data group, then `scale = FP8_MAX / amax`, multiply all values by scale before converting to FP8; dequantization simply divides by scale. This way, FP8's limited 256 representable values concentrate on the actual data range, maximizing quantization precision.

### 3.2 Three Scaling Granularities

How much data should a single scale factor cover? Larger coverage means outliers have more destructive impact on overall precision; finer coverage yields higher precision but greater overhead. This leads to three granularity levels:

```
Tensor-wise            Block-wise             Tile-wise
(1 scale / tensor)     (1 scale / 128x128)    (1 scale / 1x128)

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │ ┌───┐┌───┐┌───┐ │    │ ════════════════│
│                 │    │ │ s1││ s2││ s3│ │    │ s1 s2 s3 s4 s5  │
│   scale = s     │    │ ├───┤├───┤├───┤ │    │ ════════════════│
│                 │    │ │ s4││ s5││ s6│ │    │ s6 s7 s8 s9 ..  │
│                 │    │ └───┘└───┘└───┘ │    │ ════════════════│
└─────────────────┘    └─────────────────┘    └─────────────────┘
precision: low         precision: medium       precision: high
overhead:  minimal     overhead:  moderate     overhead:  higher
```

| Method | Granularity | Precision | Efficiency | Outlier Robustness |
|--------|------------|-----------|------------|-------------------|
| Tensor-wise | 1 scale per entire matrix | Low | Highest | Poor |
| Block-wise (128×128) | 1 scale per sub-block | Medium | High | Good |
| Tile-wise (1×128) | 1 scale per 128 elements per row | High | Medium | Best |

**Tensor-wise** was the earliest approach — simply find the maximum absolute value of the entire matrix as the scale. The problem is extreme vulnerability to outliers: a single anomalously large value (e.g., everything is at 0.01 scale but one value is 10) dominates the scale for the entire matrix, drastically reducing quantization resolution for the remaining 99.99% of normal values. This approach is now essentially obsolete.

**Block-wise** partitions the matrix into 128×128 sub-blocks, each with an independent scale. Outlier impact is confined to its own sub-block, preventing contamination of the entire matrix. Scale is stored in FP32 — for a block of 128×128 = 16,384 elements, the overhead is merely 4 bytes / 16384 ≈ 0.02%.

**Tile-wise** goes finer still, computing one scale per 128 channels along the token dimension. Activation outliers tend to concentrate in specific channels (some channels' average magnitude can be 10-100× that of others), and per-token fine-grained scaling adapts precisely to this pattern.

**DeepSeek-V3's choice** combines both: block-wise 128×128 for weights (weight distributions are stable, block granularity suffices) and tile-wise 1×128 for activations (activations have more outliers, requiring finer granularity). The result is FP8 training loss within 0.25% of the BF16 baseline [1] — demonstrating that FP8 training is entirely viable.

### 3.3 FP4: Format and Deployment Constraints

FP4 has only 16 values, requiring even finer-grained scaling strategies. Two mainstream approaches currently exist: NVIDIA's proprietary NVFP4 and the OCP open standard MXFP4:

**NVFP4 (NVIDIA proprietary)**:

```
Data layer:  [v₁ v₂ ... v₁₆]  ← groups of 16 E2M1 values
             ↕
Block scale:  s_block (E4M3, 8-bit)  ← 1 per group
             ↕
Tensor scale:  s_tensor (FP32)  ← 1 per tensor
```

**MXFP4 (OCP open standard)**:

```
Data layer:  [v₁ v₂ ... v₃₂]  ← groups of 32 E2M1 values
             ↕
Block scale:  s_block (E8M0, 8-bit, power-of-2)  ← 1 per group
```

| Property | NVFP4 | MXFP4 |
|----------|-------|-------|
| Block size | 16 | 32 |
| Scale structure | Two-level (E4M3 + FP32) | Single-level (E8M0) |
| Effective per-parameter cost | ~4.5 bits | ~4.25 bits |
| Precision | Higher | Slightly lower |
| B200 throughput | Baseline | Slightly higher (E8M0 scale is simpler in hardware) |
| Ecosystem | NVIDIA proprietary | AMD/Intel/Meta/... multi-vendor |

There is an interesting trade-off here: NVFP4 uses smaller blocks (16 vs 32) and two-level scaling for better precision. In GPT-OSS fine-tuning experiments, NVFP4's validation loss was about 2-3% lower than MXFP4 [4]. But MXFP4's E8M0 power-of-2 scale only requires bit-shift for dequantization, making it simpler in hardware with slightly better throughput; it also enjoys broader hardware support as an OCP open standard. In practice, the choice depends on precision vs ecosystem priorities.

Also worth noting is the "4.25 bits" figure — FP4 is nominally 4-bit, but with scale storage overhead, each parameter effectively costs 4.25-4.5 bits. Moreover, 4-bit is not byte-aligned, so two FP4 values must be packed into a single byte, with scales also tightly arranged:

```
1 Byte (8 bits):
┌───────────────┬───────────────┐
│  FP4 value A  │  FP4 value B  │
│   S E E M     │   S E E M     │
│   high 4 bits │   low 4 bits  │
└───────────────┴───────────────┘

NVFP4 memory layout (16 elements + 1 scale):
┌──────────────────────────────┬──────────┐
│   8 bytes (16× FP4 packed)   │  1 byte  │
│   v1,v2,v3,v4 ... v15,v16    │  scale   │
│                              │  (E4M3)  │
└──────────────────────────────┴──────────┘
```

This adds complexity to kernel implementation and memory access patterns — reading a single FP4 value requires bit shifting and masking, and scale alignment needs special handling.

On the hardware side, native FP4 Tensor Core support is currently limited to NVIDIA's Blackwell architecture (B100/B200). On Hopper and earlier GPUs, FP4 can only be implemented via software emulation, unable to achieve real throughput gains. This is the primary reason FP4 has not yet seen the widespread adoption of FP8.

---

## 4. Reducing Precision Loss

Low precision is not a free lunch. Naively switching a model from BF16 to FP8 or FP4 often leads to training loss divergence and inference quality degradation. Precision loss stems from three main sources:

- **Accumulation error**: Individual multiply errors are tiny, but accumulating thousands in a matmul amplifies them → High-precision accumulation
- **Rounding bias**: Too few representable values cause deterministic rounding to introduce systematic bias → Stochastic Rounding
- **Inter-module differences**: Different modules tolerate different precision levels; uniform low precision is both wasteful and harmful → Module-level mixed precision
- **Quantization-induced degradation**: QAT lets the model adapt to quantization during training

### 4.1 High-Precision Accumulation

Matrix multiplication is the most fundamental compute unit in deep learning. A single Linear layer's matmul may involve thousands of multiply-accumulate operations. The rounding error from a single FP8 multiplication is small, but after thousands of accumulations, the error grows to non-negligible levels.

DeepSeek-V3's approach: Tensor Cores perform FP8 multiplications for throughput, while partial sums are accumulated via CUDA Cores in FP32 precision to preserve correctness [1] — low-precision multiply, high-precision accumulate.

### 4.2 Stochastic Rounding

Accumulation error can be addressed with high-precision accumulators, but FP4 training faces another challenge: rounding bias.

Consider the value 0.2 — the two nearest FP4 representable values are 0 and 0.5. Deterministic rounding (Round-to-Nearest) always rounds it to 0, permanently losing information. If gradient updates across many parameters suffer this systematic truncation, training cannot converge.

```
Value 0.2 between FP4 representable values {0, 0.5}:
  Round-to-Nearest: always rounds to 0  → persistent information loss
  Stochastic Rounding: 40% probability → 0.5, 60% probability → 0
                       E[SR(0.2)] = 0.2  ✓ unbiased
```

Stochastic Rounding rounds probabilistically, guaranteeing that the expected value equals the original (unbiased estimation). Individual rounds are noisy, but over many iterations the parameter update direction remains correct. With FP4's mere 16 representable values, this unbiasedness is virtually a prerequisite for training convergence. Quartet, NVFP4, and other FP4 training schemes all adopt this strategy.

### 4.3 Module-Level Mixed Precision

Not all modules need equal precision. Different Transformer modules vary significantly in their sensitivity to quantization. Concentrating the precision budget on the most sensitive modules is the most straightforward engineering approach to precision protection.

| Module | Sensitivity | Recommended Precision | Key Reason |
|--------|-----------|---------------------|------------|
| Embedding | High | BF16/FP16 | Semantic representation entry point; errors propagate to all subsequent layers |
| Attention QKV | High | FP8+ | Small errors alter attention distribution, affecting semantic focus |
| Attention Score | High | BF16 | Softmax is highly sensitive to inputs; low precision causes attention collapse |
| FFN / MLP | Low | FP4/INT4 | 60-80% of parameters, high error tolerance, maximum compression benefit |
| KV Cache | High | BF16 | FP8 introduces some precision loss, affecting long-text generation quality |
| LayerNorm | High | FP32 | Very few parameters (a few KB); high precision at no extra cost |
| MoE Experts | Low | FP4/INT4 | 90%+ of total parameters, only a few activated per inference |

**Why is FFN more compression-tolerant than Attention?** FFN's parameter matrices primarily serve as "knowledge storage" — small perturbations to individual parameters are smoothed by subsequent nonlinear functions and LayerNorm. In Attention, $QK^T$ dot products feed directly into Softmax — an exponential function where tiny input perturbations are amplified exponentially. This is why DeepSeek-V4 compresses MoE Experts (essentially FFN) to FP4 while keeping Attention computation at FP8 [2].

### 4.4 Quantization-Aware Training

The preceding methods reduce error at the computation level. QAT takes a different approach from the training perspective: letting the model "see" quantized weights during training, proactively adapting to precision degradation.

DeepSeek-V4 applies FP4 (MXFP4) QAT to MoE Expert weights, with a clever core design: **FP4→FP8 dequantization is lossless** [2]. FP8 E4M3 has 2 more exponent bits than FP4 E2M1, providing greater dynamic range. As long as the scale factor ratio across FP4 sub-blocks within the same FP8 quantization block stays below a certain threshold, the fine-grained FP4 scale information can be fully absorbed by FP8's larger dynamic range. This means the entire QAT pipeline can directly reuse the existing FP8 training framework with no modifications to the backward pass.

The concrete workflow:

```
During training (simulated quantization):
FP32 master weights → quantize to FP4 → lossless dequantize to FP8 → forward computation
Backward pass: gradients pass directly from FP8 weights back to FP32 master weights (Straight-Through Estimator, STE)

During inference (real quantization):
Directly use native FP4 weights → actual speedup + memory savings
```

Simulated quantization during training preserves gradient signal integrity, while real FP4 during inference achieves actual hardware acceleration — model behavior stays consistent across both phases.

---

## 5. Summary

| Scenario | Recommended Format | Hardware Requirement | Quality Loss |
|----------|-------------------|---------------------|-------------|
| Training (default) | BF16 | Ampere+ | None |
| Training (efficient) | FP8 | Hopper+ | Minimal, near-lossless |
| Training (aggressive) | FP4 | Blackwell | Requires QAT |
| Inference (near-lossless) | FP8 | Hopper+ | Negligible |
| Inference (memory-saving) | INT4 AWQ/GPTQ | Universal | Slight degradation |
| Inference (max throughput) | NVFP4 / MXFP4 | Blackwell | Requires QAT |

Low-precision computation is not simply about reducing bit width — it requires systematic design across format selection, scaling strategy, and module-level precision allocation. From DeepSeek-V3's FP8 to DeepSeek-V4's FP4+FP8 hybrid to GPT-OSS's native MXFP4 release, this trajectory is maturing rapidly.

FP8's 256 representable values are far from precise enough on their own — fine-grained scaling, high-precision accumulation, and other engineering techniques are what make training loss converge. The same applies to FP4: QAT, Stochastic Rounding, and module-level mixed precision are each insufficient alone, but combined they make 4-bit training work. As Blackwell's native FP4 hardware support matures, FP4 may well become the new default for both large model training and inference within the next year or two.

---

## References

[1] DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437, 2024. https://arxiv.org/abs/2412.19437

[2] DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." 2026. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf

[3] OpenAI. "Introducing GPT-OSS." 2025. https://openai.com/index/introducing-gpt-oss/

[4] NVIDIA. "Fine-Tuning GPT-OSS for Accuracy and Performance with Quantization Aware Training." NVIDIA Developer Blog, 2025. https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/

[5] Micikevicius et al. "Mixed Precision Training." ICLR 2018. https://arxiv.org/abs/1710.03740
