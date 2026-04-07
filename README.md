![AI Accelerator Architecture](https://img.shields.io/badge/topic-AI%20accelerator%20architecture-blue)

# AI Accelerator Architecture -- Interview Preparation

A comprehensive study guide covering the architecture of hardware accelerators for AI and machine learning workloads. This repository is designed for engineers preparing for interviews at companies building GPUs, TPUs, custom ASICs, and other domain-specific accelerators for deep learning training and inference.

---

## Table of Contents

### 1. Foundations
Core concepts explaining why AI accelerators exist and the fundamental metrics used to evaluate them.

- [Why AI Accelerators](01_foundations/why_ai_accelerators.md) -- Dennard scaling, Moore's law slowdown, domain-specific architectures
- [Compute Fundamentals](01_foundations/compute_fundamentals.md) -- FLOPS, TOPS, ops/watt, arithmetic intensity, roofline model
- [Workload Characteristics](01_foundations/workload_characteristics.md) -- CNNs, transformers, GEMMs, convolution as GEMM
- [Worked Problems: Foundations](01_foundations/worked_problems/)
  - [Problem 1: Roofline Analysis](01_foundations/worked_problems/problem_01_roofline_analysis.md)
  - [Problem 2: Ops per Watt Comparison](01_foundations/worked_problems/problem_02_ops_per_watt_comparison.md)
  - [Problem 3: Workload Profiling](01_foundations/worked_problems/problem_03_workload_profiling.md)

### 2. Architecture Paradigms
The major architectural approaches to building AI accelerators.

- [GPU Architecture](02_architecture_paradigms/gpu_architecture.md) -- CUDA cores, tensor cores, SMs, warp scheduling, A100/H100/B200
- [Systolic Arrays and TPUs](02_architecture_paradigms/systolic_arrays_and_tpus.md) -- Matrix multiply units, bfloat16, Google TPU v4/v5
- [Dataflow Architectures](02_architecture_paradigms/dataflow_architectures.md) -- Spatial architectures, dataflow taxonomies
- [Worked Problems: Architecture Paradigms](02_architecture_paradigms/worked_problems/)
  - [Problem 1: Systolic Array Mapping](02_architecture_paradigms/worked_problems/problem_01_systolic_array_mapping.md)
  - [Problem 2: GPU vs TPU Tradeoffs](02_architecture_paradigms/worked_problems/problem_02_gpu_vs_tpu_tradeoffs.md)
  - [Problem 3: Dataflow Scheduling](02_architecture_paradigms/worked_problems/problem_03_dataflow_scheduling.md)

### 3. Memory Hierarchy
Understanding the memory wall and how accelerator architects address bandwidth limitations.

- [Memory Wall and Bandwidth](03_memory_hierarchy/memory_wall_and_bandwidth.md) -- Bandwidth bottleneck, data reuse, tiling strategies
- [On-Chip Memory Design](03_memory_hierarchy/on_chip_memory_design.md) -- SRAM banks, multi-ported memories, double buffering
- [HBM and Off-Chip Memory](03_memory_hierarchy/hbm_and_off_chip_memory.md) -- Die stacking, channel architecture, HBM2e/HBM3/HBM3e
- [Worked Problems: Memory Hierarchy](03_memory_hierarchy/worked_problems/)
  - [Problem 1: Memory Bandwidth Calculation](03_memory_hierarchy/worked_problems/problem_01_memory_bandwidth_calculation.md)
  - [Problem 2: SRAM vs HBM Partitioning](03_memory_hierarchy/worked_problems/problem_02_sram_vs_hbm_partitioning.md)
  - [Problem 3: Data Reuse Optimization](03_memory_hierarchy/worked_problems/problem_03_data_reuse_optimization.md)

### 4. Datapath and Compute
The arithmetic engines at the heart of every accelerator.

- [MAC Units and Precision](04_datapath_and_compute/mac_units_and_precision.md) -- INT8, FP16, BF16, FP8, TF32 formats
- [Quantization and Mixed Precision](04_datapath_and_compute/quantization_and_mixed_precision.md) -- PTQ, QAT, GPTQ, AWQ, mixed-precision training
- [Sparsity Acceleration](04_datapath_and_compute/sparsity_acceleration.md) -- Structured vs unstructured, 2:4 sparsity, compressed sparse formats
- [Worked Problems: Datapath and Compute](04_datapath_and_compute/worked_problems/)
  - [Problem 1: MAC Array Design](04_datapath_and_compute/worked_problems/problem_01_mac_array_design.md)
  - [Problem 2: Quantization Impact](04_datapath_and_compute/worked_problems/problem_02_quantization_impact.md)
  - [Problem 3: Sparse Matrix Engine](04_datapath_and_compute/worked_problems/problem_03_sparse_matrix_engine.md)

### 5. Interconnect and NoC
On-chip and off-chip communication for scaling accelerators.

- [On-Chip Interconnect](05_interconnect_and_noc/on_chip_interconnect.md) -- Buses, crossbars, flow control mechanisms
- [NoC Topologies](05_interconnect_and_noc/noc_topologies.md) -- Mesh, ring, tree, crossbar, bandwidth provisioning
- [Multi-Chip Scaling](05_interconnect_and_noc/multi_chip_scaling.md) -- NVLink, chip-to-chip interconnect, wafer-scale, scale-out
- [Worked Problems: Interconnect and NoC](05_interconnect_and_noc/worked_problems/)
  - [Problem 1: NoC Bandwidth Design](05_interconnect_and_noc/worked_problems/problem_01_noc_bandwidth_design.md)
  - [Problem 2: Chip-to-Chip Interconnect](05_interconnect_and_noc/worked_problems/problem_02_chip_to_chip_interconnect.md)
  - [Problem 3: Collective Communication](05_interconnect_and_noc/worked_problems/problem_03_collective_communication.md)

### 6. System and Software
The software stack, power delivery, and system-level design considerations.

- [Compiler and Mapping](06_system_and_software/compiler_and_mapping.md) -- XLA, TVM, Triton, operator fusion, tiling, scheduling
- [Power and Thermal Management](06_system_and_software/power_and_thermal_management.md) -- TDP, power delivery for 700W+ chips, thermal solutions
- [Training vs Inference Architectures](06_system_and_software/training_vs_inference_architectures.md) -- Batch size, latency vs throughput, KV cache, speculative decoding
- [Worked Problems: System and Software](06_system_and_software/worked_problems/)
  - [Problem 1: Model Partitioning](06_system_and_software/worked_problems/problem_01_model_partitioning.md)
  - [Problem 2: Power Envelope Design](06_system_and_software/worked_problems/problem_02_power_envelope_design.md)
  - [Problem 3: Inference Latency Optimization](06_system_and_software/worked_problems/problem_03_inference_latency_optimization.md)

### 7. Quizzes
Multiple-choice quizzes to test your knowledge across all topics.

- [Quiz: Foundations](07_quizzes/quiz_foundations.md)
- [Quiz: Architecture](07_quizzes/quiz_architecture.md)
- [Quiz: Memory and Compute](07_quizzes/quiz_memory_and_compute.md)
- [Quiz: System](07_quizzes/quiz_system.md)

---

## How to Use

1. **Study sequentially** -- Start with Section 1 (Foundations) and progress through each section in order. Later sections build on earlier concepts.
2. **Attempt worked problems** -- After reading the concept files in each section, attempt the worked problems before looking at the solutions.
3. **Take quizzes** -- Use the quizzes in Section 7 to test retention after completing each major topic area.
4. **Cross-reference** -- Follow the relative links between files to deepen understanding of related topics.
5. **Review before interviews** -- Focus on the Q&A format entries as they mirror common interview question patterns.

---

## Contributing

Contributions are welcome. Please open a pull request with your proposed changes. Ensure that:

- Content follows the existing formatting conventions (no emoji, Q&A format for concept files)
- All new files are linked from this README
- Worked problems include both a Problem Statement and a Worked Solution with numbered steps
- Quiz questions include an answer key

---

## Related Repositories

- [Interview Preparation -- Digital Design](https://github.com/BrendanJamesLynskey/Interview_Digital_Design)
- [Interview Preparation -- VLSI Physical Design](https://github.com/BrendanJamesLynskey/Interview_VLSI_Physical_Design)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Last updated: 2026-04-07
