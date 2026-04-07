# Power and Thermal Management

This section covers TDP, power delivery challenges for high-power AI chips, thermal solutions, and power capping strategies.

---

### Q1. What is TDP and how does it relate to actual power consumption?

**Answer:**

TDP (Thermal Design Power) is the maximum sustained power that a chip is designed to dissipate under worst-case workloads. It defines the thermal solution requirements: the cooling system must be capable of removing at least TDP watts of heat from the chip.

TDP is not the maximum instantaneous power draw, which can exceed TDP during short transients. Modern AI chips experience power spikes of 1.2-1.5x TDP during burst workloads (e.g., a large GEMM starting up with full tensor core utilization). The power delivery network must handle these spikes without voltage droop that could cause timing violations.

TDP has increased dramatically for AI accelerators: A100 at 400W, H100 at 700W, B200 at ~1000W, AMD MI300X at 750W. This trend is driven by the demand for more compute throughput: increasing MAC unit count, higher clock frequencies, and wider memory interfaces all increase power.

For data center planning, the relevant metric is the average power consumption under the expected workload mix, which is typically 60-90% of TDP for training workloads (tensor cores active most of the time) and 40-70% for inference workloads (variable utilization depending on batching and model characteristics).

---

### Q2. What are the power delivery challenges for 700W+ AI chips?

**Answer:**

Delivering 700-1000W to a single chip creates several engineering challenges:

**Current magnitude**: At a core voltage of approximately 0.75V, a 700W chip draws nearly 1000A of current. The power delivery network (PDN) from the voltage regulator modules (VRMs) through the PCB, package, and on-chip power grid must carry this current without excessive resistive losses (I^2*R heating) or voltage drop (IR drop).

**Voltage droop**: When the chip transitions from idle to full load (or between different workload phases), the current demand changes rapidly (di/dt). The inductance in the PDN creates voltage transients (V = L * di/dt) that can cause the core voltage to drop below the minimum required for correct operation. Decoupling capacitors at multiple levels (PCB, package, on-die) provide local charge reservoirs to mitigate droop.

**Power integrity**: The on-chip power grid must distribute current uniformly across the die. Hot spots (regions with higher current draw) can cause localized voltage drop, creating timing violations in those regions. The power grid is typically implemented as a dense mesh in the top metal layers, consuming 5-10% of the metal routing resources.

**VRM design**: The VRMs must deliver 1000A at high efficiency (>90%) with fast transient response. Multi-phase buck converters with 12-16 phases are typical for high-power AI accelerators. The VRMs are placed close to the chip on the server board to minimize trace inductance.

**Connector and PCB design**: The power must be delivered from the server's power supply unit (PSU) through connectors (12V or 48V) and PCB traces to the VRMs. At 12V, 700W requires 58A from the connector, which may exceed the current capacity of standard PCIe power connectors (150-225W each). High-power AI cards use multiple auxiliary power connectors or custom high-current connectors. The transition to 48V distribution (instead of 12V) reduces current by 4x, easing connector and PCB requirements.

---

### Q3. How do modern AI chips manage thermal dissipation?

**Answer:**

Thermal management for AI accelerators has evolved from air cooling to liquid cooling as TDPs have increased:

**Air cooling (up to ~350W)**: A heatsink with high-surface-area fins is attached to the chip via thermal interface material (TIM). Fans force air through the heatsink fins, carrying away heat. The thermal resistance from chip to ambient determines the chip temperature. At 350W with a high-performance heatsink, the chip temperature reaches 80-90C.

**Liquid cooling (350-1000W+)**: A cold plate carrying liquid coolant (typically water or water-glycol mixture) is attached to the chip. The liquid absorbs heat and carries it to a heat exchanger (either within the rack or at a facility level). Liquid cooling provides 5-10x better heat transfer than air cooling, enabling higher TDPs in the same volume.

**Direct-to-chip liquid cooling**: The cold plate contacts the chip's heat spreader directly. This is the most common liquid cooling approach for AI accelerators. NVIDIA's DGX H100 systems use direct-to-chip liquid cooling.

**Immersion cooling**: The entire server board is submerged in a dielectric fluid (non-conductive liquid). This provides uniform cooling across all components (chip, VRMs, memory, PCB) and eliminates the need for fans and heatsinks. It is gaining adoption for high-density AI installations.

**Thermoelectric cooling (TEC)**: Peltier devices can provide localized cooling below ambient temperature but consume additional power (typically 50-100% of the heat being pumped). Used in extreme cases where specific chip regions need to be kept at very low temperatures.

The choice of cooling technology directly affects data center design. Air-cooled racks are limited to approximately 30-50 kW per rack. Liquid-cooled racks can support 70-120 kW or more. Immersion-cooled systems can achieve even higher densities.

---

### Q4. What is dynamic voltage and frequency scaling (DVFS) and how is it used in AI chips?

**Answer:**

DVFS adjusts the operating voltage and clock frequency of the chip in real time based on workload demands and thermal conditions. Reducing voltage saves power quadratically (P proportional to V^2 * f) while reducing frequency linearly impacts performance.

In AI accelerators, DVFS is used for: (1) power capping -- reducing frequency to stay within a power limit when the workload exceeds the thermal solution's capacity, (2) thermal throttling -- reducing frequency when the chip temperature exceeds safe limits, (3) boost clocking -- increasing frequency above the base clock when power and thermal headroom exist, and (4) idle power reduction -- aggressively reducing voltage and frequency during idle periods.

NVIDIA's GPU Boost technology dynamically adjusts the tensor core and memory clock frequencies based on the chip's power consumption, temperature, and the number of active SMs. Under favorable conditions (good cooling, low-power workload), the boost clock can be 10-20% above the base clock. Under unfavorable conditions (thermal throttling, power-limited), the clock drops to ensure the chip stays within its operating envelope.

For AI workloads, DVFS behavior affects benchmark consistency. Two identical GPUs may achieve different training throughput depending on cooling quality (which affects the sustained boost clock). Enterprise-grade systems (DGX) provide consistent cooling to maintain maximum boost clocks under sustained load.

---

### Q5. What is power capping and why is it important for data center deployments?

**Answer:**

Power capping limits the maximum power a chip or server can consume, regardless of workload demand. The chip's power management controller monitors real-time power consumption and reduces clock frequency and/or voltage to stay within the cap.

Power capping is important for several reasons:

**Infrastructure protection**: Data center power infrastructure (PDUs, breakers, busbars, transformers) has fixed capacity. If all servers simultaneously draw their maximum rated power, the infrastructure could be overloaded. Power capping ensures that the aggregate power stays within infrastructure limits.

**Over-provisioning**: By setting power caps below the TDP, data center operators can deploy more servers than the infrastructure could support at full TDP, knowing that not all servers will hit TDP simultaneously. This "power over-provisioning" improves infrastructure utilization.

**Cost optimization**: Electricity costs increase non-linearly at very high power (due to demand charges and cooling costs). Capping power at a level that provides the best performance-per-watt (which is below maximum power) can reduce total cost of ownership.

**Workload management**: Different AI workloads have different power-performance tradeoff curves. Training a large model (compute-bound, high tensor core utilization) uses near-TDP power. Running batch inference (memory-bound, low compute utilization) may use only 50-60% of TDP. Power capping can allocate power budget from idle or underutilized servers to heavily loaded ones.

In practice, NVIDIA GPUs support power capping through the nvidia-smi interface (nvidia-smi --power-limit=X). The GPU's internal power management controller enforces the limit by adjusting clock frequency. A 10% power reduction typically causes a 3-7% performance reduction, demonstrating the non-linear relationship between power and performance.

---

### Q6. How does chip power break down between compute, memory, I/O, and leakage?

**Answer:**

For a modern AI accelerator (approximate breakdown for H100 at 700W TDP):

| Component | Power (W) | Fraction |
|---|---|---|
| Tensor cores + CUDA cores | 250-350 | 36-50% |
| HBM (off-chip DRAM) | 50-75 | 7-11% |
| On-chip SRAM (L2 + shared mem) | 40-60 | 6-9% |
| NoC and interconnect | 30-50 | 4-7% |
| I/O (NVLink, PCIe, HBM PHY) | 50-80 | 7-11% |
| Clock distribution | 40-60 | 6-9% |
| Leakage (static power) | 80-120 | 11-17% |
| Control logic, misc | 30-50 | 4-7% |

Key observations:
- The compute units (tensor cores) consume 36-50% of total power. The remaining 50-64% goes to "overhead" -- moving data, distributing clocks, and leakage. This highlights that data movement, not computation, dominates energy consumption in modern accelerators.
- Leakage power (11-17%) is significant and cannot be eliminated through design optimization alone -- it is a property of the transistor technology. Leakage increases with temperature, creating a positive feedback loop: more leakage -> more heat -> higher temperature -> more leakage.
- HBM power (7-11%) is relatively modest but represents a hard limit on memory bandwidth (more bandwidth requires more HBM stacks and channels, each consuming additional power).

For energy efficiency optimization, the largest opportunities lie in reducing data movement energy: keeping data in registers and local SRAM (low-energy access) rather than fetching from HBM (high-energy access), and using operator fusion and tiling to maximize data reuse.

---

### Q7. What thermal management challenges are unique to multi-die/chiplet designs?

**Answer:**

Multi-die designs (like AMD MI300X with 8 active dies and NVIDIA B200 with 2 compute dies) introduce thermal challenges beyond those of monolithic designs:

**Non-uniform heat generation**: Different chiplets have different power densities. Compute chiplets may dissipate 100-200 W/cm^2, while I/O chiplets operate at 20-50 W/cm^2. The cooling solution must handle this non-uniformity without over-cooling the low-power dies or under-cooling the high-power dies.

**Thermal coupling**: Multiple hot dies in close proximity heat each other through the shared package substrate and heat spreader. The temperature of one die affects its neighbors, creating thermal coupling that complicates thermal management. A die in the center of a multi-die package runs hotter than an edge die because it is surrounded by heat sources.

**HBM thermal sensitivity**: HBM stacks are temperature-sensitive (JEDEC specifies maximum 95C). In designs where HBM stacks are adjacent to hot compute dies on an interposer, the HBM temperature is influenced by the compute die's heat. The cooling solution must ensure HBM stays within limits even when compute dies are at full power.

**Uneven cooling**: A single cold plate or heatsink may not provide uniform cooling across all dies in a multi-die package. Die height variations (due to different die thicknesses and bonding stack-ups) create uneven contact with the cold plate. Thermal interface materials must accommodate these variations.

**Power map changes**: Different workloads activate different chiplets to different degrees. The thermal solution must handle all possible power distribution scenarios, including the worst case where all chiplets are at maximum power.

Solutions include: separate thermal sensors per chiplet (enabling per-chiplet throttling), improved TIM materials (filling gaps between dies and the heat spreader), custom cold plate designs with targeted flow patterns, and vapor chamber heat spreaders that equalize temperature across the package.

---

### Q8. How does the trend toward higher TDP impact data center design?

**Answer:**

The increase from 300W (V100, 2017) to 700W (H100, 2022) to 1000W+ (B200, 2024) per chip has profound implications for data center infrastructure:

**Power density**: A DGX H100 system with 8 H100 GPUs consumes approximately 10 kW. A rack with 4 DGX systems draws 40+ kW. Traditional air-cooled data centers support 10-20 kW per rack. AI deployments require 40-120 kW per rack, necessitating upgraded electrical distribution, new cooling infrastructure, and potentially new building designs.

**Cooling infrastructure**: Liquid cooling becomes mandatory at 700W+ per chip. This requires chilled water piping throughout the data center, coolant distribution units (CDUs) on or near each rack, and leak detection and containment systems. The capital cost for liquid cooling infrastructure is significant (potentially $1000-3000 per kW of cooling capacity).

**Power distribution**: Higher rack power requires larger power distribution units (PDUs), thicker power cables, and potentially higher voltage distribution (48V rack-level DC instead of 12V) to reduce cable losses and thickness. Some facilities are moving to 48V DC distribution or even medium-voltage (20kV) distribution to the rack level.

**Facility electrical capacity**: A large AI training cluster (10,000+ GPUs at 700W each) requires 7+ MW of IT power plus 2-3 MW of cooling, totaling 10+ MW -- comparable to a small power plant. Securing this power capacity and the associated grid connections can take years and is a key constraint on AI compute scale-up.

**Stranded capacity**: Many existing data centers were designed for lower power densities and cannot support AI workloads without expensive retrofitting. This has created demand for purpose-built AI data centers.

---

### Q9. What is the energy cost of training a large language model?

**Answer:**

The energy cost can be estimated from the training time, number of GPUs, and per-GPU power:

For GPT-4-scale training (estimated 25,000 A100 GPU-equivalents for ~100 days):
```
Energy = 25,000 GPUs * 400W * 100 days * 24 hours/day
       = 25,000 * 0.4 kW * 2400 hours
       = 24,000,000 kWh = 24 GWh
```

At $0.08/kWh electricity cost:
```
Electricity cost = 24 GWh * $0.08/kWh = $1.92 million
```

Including cooling overhead (PUE 1.3):
```
Total energy = 24 * 1.3 = 31.2 GWh
Total electricity cost = $2.5 million
```

For context: 31.2 GWh is roughly the annual electricity consumption of 2,800 US households.

The total cost of training (including hardware depreciation, networking, storage, and engineering) is much higher than the electricity cost alone. For GPT-4-scale training, total cost estimates range from $50-100 million, of which electricity is 2-5%.

As models scale further and hardware becomes more power-hungry, the energy cost is growing. Future models trained on 100,000+ H100-equivalent GPUs could consume 100+ GWh per training run. This has implications for energy policy, carbon emissions, and the geographic placement of AI training facilities (near cheap, clean power sources).

---

### Q10. How do accelerator architects optimize for energy efficiency (FLOPS/W)?

**Answer:**

Energy efficiency optimization spans multiple design levels:

**Transistor level**: Choosing the optimal process node (smaller transistors have lower switching energy but higher leakage), optimizing transistor sizing (minimum-size transistors for non-critical paths to reduce capacitance), and using low-voltage design techniques (near-threshold computing for non-critical logic).

**Circuit level**: Clock gating (disabling the clock to unused logic blocks to eliminate dynamic switching power), power gating (disconnecting unused blocks from the power supply to eliminate leakage), and multi-voltage domains (running time-critical paths at higher voltage and non-critical paths at lower voltage).

**Architecture level**: Specialization (using fixed-function MAC arrays instead of general-purpose ALUs eliminates instruction fetch/decode energy), reduced precision (INT8 MACs are ~10x more energy-efficient than FP32), data reuse (minimizing memory access energy through tiling and fusion), and spatial dataflow (moving data along short wires between adjacent PEs instead of through a multi-level cache hierarchy).

**System level**: Operating at the optimal voltage-frequency point (where FLOPS/W is maximized, typically below the maximum frequency), using the right precision for each operation (FP8 for forward pass, BF16 for backward pass, FP32 for accumulation), and batching inference requests to improve compute utilization.

**Workload level**: Model architecture choices that improve ops/parameter (e.g., MoE models that activate only a fraction of parameters), pruning and quantization (reducing the work per inference), and distillation (creating smaller models with similar quality).

The energy efficiency frontier improves at approximately 2x per 3 years, combining process technology improvements (1.3x per node) with architectural improvements (1.5x per generation). Sustaining this rate of improvement is essential for the economic viability of ever-larger AI models.
