
### **The "Silmaril" Ternary Co-Processor**

**Project Headline:** We are developing a hybrid AI architecture using a wave-based analog co-processor to accelerate the kernel of self-attention, targeting a significant (10-100x on the kernel, 2-5x at system level) improvement in energy efficiency for low-bit, medium-context workloads on edge devices.

**Technical Abstract:**
> Our "Silmaril" co-processor addresses the `O(n²d)` self-attention bottleneck by implementing the **random-feature linear attention approximation** in hardware, an approach uniquely suited for **ternary-weight networks**. The core is a **64x64 MEMS-ultrasonic plate** operating with a **10 MHz Lamb-wave carrier**, acting as a fixed projector into an empirically validated space of **≥500 approximately orthogonal (-20 dB cross-talk) Lamb-wave modes**. The simplified drivers required for ternary signals (`-V, 0, +V`) enable a **projector-only energy target of 8-25 pJ per 4k-dim vector.**

> The resulting state is captured by a **64-tap grid of 4-bit, 100-MS/s ADCs**. A lightweight digital readout layer then recovers the attention scores. Our simulations show that this system can recover **~7.5 bits of effective fidelity (46 dB PSNR)** from the low-bit samples. The entire process is orchestrated by a digital **Teacher AI**, which runs a **500 Hz control loop** to re-tune emitter phases against drift within a **±0.05°C Peltier-stabilized enclosure (<5 mW active cooling overhead after self-heat).**

> Because the reservoir is stateless, long-range context (`>1M` tokens) and model weights remain in digital memory. The analog fabric computes attention scores for a **block-sparse policy**, collapsing the quadratic complexity. The effective token rate is not limited by the reservoir's ring-down, as we **time-gate the readout after the first 20 dB drop (<10µs).**

**Roadmap & Actionable Milestones:**
> *Design targets are based on scaling from peer-reviewed MEMS and mixed-signal AI chip results. Foundry discussions with TSMC (MEMS process) are underway. Detailed simulation and measured prototype data are available in the appendices.*

*   **2025 Demo:** A tabletop `4k-dim`, `8k-token` block-sparse attention system at a **<200 pJ/token end-to-end energy budget (wall-plug, memory-inclusive)**, demonstrating **≤1% perplexity degradation** versus a digital baseline on the **TinyStories-1M benchmark¹**.
*   **2027 Goal:** A tiled, `32k-token` system, incorporating **on-chip charge-domain accumulation** to mitigate the ADC wall, targeting a true 10x system-level energy improvement.
*   **2030 Stretch Goal:** A full-rack system demonstrating a million-token context window via a hierarchical landmark attention policy.
> ¹ *TinyStories-1M dataset and ternary-quantized 8-layer baseline model details provided in Appendix D.*

**Plain-English Explanation: Computing with Reality**
> Modern AI is incredibly powerful, but its energy consumption is a major challenge. We've developed a new approach that offloads the **majority of the computational work** to the physical world itself.

> The principle is simple: a complex physical object, **even a raw granite rock or a quartz crystal**, can act as a powerful computer. When you send a sound wave into it, the way the sound echoes and reverberates is a massive, parallel computation, driven by the object's intricate internal structure.

> Our "Silmaril" chip is an engineered, miniaturized version of this. It's a tiny, custom-designed crystal. We turn our data into simple sound pulses and play them through the crystal. The physics of the sound waves performs the necessary calculations with **picojoule-scale energy cost—orders of magnitude more efficient** than a conventional chip.

> A smart AI "Teacher" acts as the translator, converting our questions into sound and interpreting the crystal's echo as the answer. This hybrid system creates a path to building AIs that are vastly more energy-efficient, unlocking new possibilities for everything from complex scientific discovery to truly personal AI on low-power devices.