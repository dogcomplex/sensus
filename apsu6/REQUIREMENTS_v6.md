### **Project Apsu v6.0: A Universal Framework for Emulating Correlated Systems**
### **Engineering Requirements & Design Specification**

**Document Version:** 6.0 (Definitive Research Program)
**Date:** July 19, 2025
**Authors:** User & Assistant
**Status:** Approved for Implementation

---

### **Table of Contents**

*   **Part 1: Project Mandate & Core Scientific Background**
    1.  Research Mandate & Evolved Hypothesis
    2.  Core Scientific & Technical Background
*   **Part 2: The Universal Experimental Framework**
    3.  The Three Primary Experimental Protocols
    4.  The Universal Control Parameters & Measurement
*   **Part 3: Detailed Component Specifications & Implementation**
    5.  The `ClassicalSubstrate` Module
    6.  The `UniversalController` Module
    7.  The `ExperimentHarness` Module
*   **Part 4: The Research Program & Protocols**
    8.  The Multi-Track Research Program
    9.  Data Management, Reproducibility, and Publication Strategy
    10. Glossary of Terms
*   **Part 5: Appendices**
    *   Appendix A: Configuration Schema
    *   Appendix B: FLOPs Accounting

---

### **PART 1: PROJECT MANDATE & CORE SCIENTIFIC BACKGROUND**

---

### **1. Research Mandate & Evolved Hypothesis**

#### **1.1. High-Level Objective: From "If" to "How Much"**

This document outlines the definitive research program for Project Apsu v6.0. The initial phases of this project successfully demonstrated a computational proof-of-concept: a purely classical dynamical system, when guided by a non-local, intelligent controller, can be trained to produce statistical correlations that violate the established bounds of classical physics. The foundational question of *"if this is possible"* within a simulated environment has been answered affirmatively.

The mandate for Project Apsu v6.0 is to move beyond this existence proof into the realm of rigorous, quantitative science. Our objective is no longer to simply achieve a binary success/failure but to **systematically map the landscape of possibilities.** We now ask the far more precise and meaningful question:

**"What are the fundamental computational and informational resources required for a classical system to emulate the statistical properties of post-classical physical theories?"**

This program will transform our findings from a computational curiosity into a powerful framework for understanding the relationship between information, causality, and the apparent laws of nature. We will build a universal emulator capable of simulating different "realities" and measure the precise computational "cost" of each.

#### **1.2. The Layered-Time Hypothesis: The Core Concept**

The scientific foundation of this project is the **Layered-Time Hypothesis**. For an engineer or computer scientist unfamiliar with the intricacies of quantum foundations, this hypothesis can be understood through a concrete computational analogy:

*   **The "Slow Medium" (`S_slow`):** Imagine a complex, high-resolution physics simulation of a fluid, like a weather model. The state of this world updates at a perceivable rate, for instance, one frame per second. The "inhabitants" of this world can only observe events and receive information at a speed limited by the "speed of sound" within their fluid dynamics. This is their **characteristic timescale, `τ_slow`**.

*   **The "Fast Medium" (`S_fast`):** This is the supercomputer cluster running the weather simulation. Its processors operate on a nanosecond timescale. Between each single one-second frame experienced by the "slow medium," the supercomputer can execute billions or trillions of computational operations. This is its **characteristic timescale, `τ_fast`**.

*   **The Speed Ratio (`R`):** The ratio of these timescales, `R = τ_slow / τ_fast`, is immense.

*   **The Hypothesis:** The Layered-Time Hypothesis posits that this supercomputer (`S_fast`) could use its vast speed and computational advantage to do more than just simulate the local laws of fluid dynamics. It could observe the *entire* state of the fluid at once (a non-local view) and compute a complex, globally-coordinated outcome. It could then "paint" this outcome onto the next frame. To the inhabitants of the fluid world, two distant events could appear to be instantaneously correlated in a way that violates their known "speed of sound." They would be forced to conclude that their reality is governed by "spooky," non-local laws.

Project Apsu is an attempt to build a working, quantifiable model of this principle. We are investigating whether the "weirdness" of quantum mechanics might be an artifact of our own reality being such a "slow medium" relative to a hidden, faster computational layer.

#### **1.3. The Four Worlds Model: A Taxonomy of Correlation**

Our research is now framed by a four-tiered model of computational reality. This taxonomy provides a structured way to understand our results. Each "world" is defined by the fundamental physical principles it obeys, which in turn sets a hard mathematical limit on the correlations it can produce. We will measure these correlations using the **Bell-CHSH `S`-score**.

1.  **The Classical World (`S_max = 2`):**
    *   **Governing Principle:** Local Realism. All information is "real" (properties exist before measurement) and "local" (no influence can travel faster than a fixed speed limit).
    *   **Description:** This is the intuitive world of everyday experience and Newtonian physics.

2.  **The Quantum World (`S_max = 2√2 ≈ 2.828`):**
    *   **Governing Principles:** Quantum Mechanics. This world abandons local realism. Its state is described by a wavefunction in a Hilbert space. Its evolution is **unitary** (information-preserving), and it is constrained by the **Non-Signaling Principle**.
    *   **Description:** This is the world our universe appears to inhabit, with its characteristic entanglement and probabilistic measurements. The `S ≤ 2.828` limit is known as **Tsirelson's Bound**.

3.  **The Mannequin World (PR-Box, `S_max = 4`):**
    *   **Governing Principles:** A Non-Local, Super-Deterministic, but still Non-Signaling Theory. This world is governed by a controller that has a global view of the system, allowing it to create correlations stronger than quantum mechanics permits. However, it is still constrained by causality in a way that prevents it from being used for faster-than-light communication.
    *   **Description:** This is our "Mannequin Universe" in its most powerful form. It corresponds to a theoretical object known as a **Popescu-Rohrlich (PR) Box**.

4.  **The Absurdist World (Signaling, `S > 4`):**
    *   **Governing Principles:** A Non-Local, Signaling Theory. This world abandons the Non-Signaling principle. The controller is explicitly allowed to use information from one part of the system to influence the immediate output of another part.
    *   **Description:** This is a "broken" experiment from a physics perspective, as it allows for direct communication. However, from a computer science perspective, it is a system with a measurable communication channel, whose properties we can study.

Project Apsu v6.0 will build emulators for the latter three worlds to understand the precise computational resources required to access each level of this hierarchy.

#### **1.4. Primary Research Goal: Mapping the Resource Trade-offs**
The primary deliverable of this project will be a series of empirical "scaling laws" that function as a "phase diagram" for these computational worlds. We aim to produce plots and, eventually, functions that answer questions like:

> "To achieve a target correlation strength `S`, what is the minimum required controller speed-ratio `R`, information-ratio `K`, and sensor precision (inverse of `σ_noise`)?"

This moves the project from philosophical speculation to quantitative, predictive science.

---

### **2. Core Scientific & Technical Background**

This section provides the necessary background for an engineer to understand the design choices made in this document without requiring prior expertise in quantum foundations or advanced machine learning.

#### **2.1. A Practical Introduction to Reservoir Computing (RC)**

Reservoir Computing is a computational paradigm that provides an ideal, simplified model for the "slow medium" in our experiment. It is a form of recurrent neural network that embraces randomness and complex dynamics.

*   **2.1.1. What is a Reservoir?**
    *   Imagine a network of `N` interconnected "neurons" (nodes). The connections between them are represented by a large `N x N` weight matrix, `W`.
    *   **The Key Idea:** Unlike traditional neural networks, these internal connections `W` are **not trained**. They are generated randomly at the beginning and then are held **permanently fixed**.
    *   This fixed, random network is the "reservoir." It is a high-dimensional, non-linear dynamical system.

*   **2.1.2. How does it Compute?**
    *   **Excitation & Dynamics:** An input signal `u(t)` is fed into the reservoir. This signal acts as an external driving force, perturbing the system. The state of the reservoir `x(t)` evolves according to a simple, deterministic rule that depends on its previous state `x(t-1)` and the current input `u(t)`. The standard update equation is the **leaky-integrator ESN equation**:
        `x(t+1) = (1 - a) * x(t) + a * tanh(W * x(t) + W_in * u(t+1))`
        Where `W_in` is a fixed random input matrix and `a` is a "leaking rate" that controls the system's memory.
    *   **Feature Extraction:** The crucial effect of this process is that the internal state vector `x(t)` becomes a rich, complex, and unique "fingerprint" of the recent history of the input signal `u`. The reservoir acts as an automatic, non-linear feature extractor, projecting the simple input into a much higher-dimensional space where complex patterns become easier to identify. This computation is "free," a natural consequence of the system's physics.
    *   **Learning:** The only part of the system that learns is a simple, linear "readout" layer. This readout is a matrix `W_out` that is trained (usually with one-shot linear regression) to map the complex reservoir state `x(t)` to a desired output `y(t)`.
        `y(t) = W_out * x(t)`

*   **2.1.3. Why is it the Right Tool for this Project?**
    *   **Analog to Physics:** The ESN model is an excellent abstract representation of a real-world physical dynamical system, like a turbulent fluid, a biological neural circuit, or our proposed "glass cubes."
    *   **Controllable Complexity:** The "chaoticity" of the reservoir can be precisely tuned via its hyperparameters (like the spectral radius `sr`), allowing us to create a substrate with the exact level of complexity we need.
    *   **"Black-Box" Substrate:** Because its internal workings are random and fixed, it's a perfect model of a system whose internal logic is unknown. Our `UniversalController` must learn to control it based only on its observable behavior, just as we must do with real physical systems.
    *   **Implementation:** We will use the `reservoirpy` Python library, a mature framework for building and experimenting with these systems.

#### **2.2. The Bell-CHSH Game: The Project's Universal Metric**

To objectively measure the "quantumness" of our system, we use a standard tool from quantum foundations: the CHSH game. An engineer should treat this as a rigorous, black-box performance test.

*   **2.2.1. The Rules of the Game**
    1.  **Systems:** Two separated systems, A and B (our two ESNs).
    2.  **Inputs (The "Challenge"):** In each round of the game, a referee provides each system with a random binary input setting. Alice's system `A` receives setting `a ∈ {0, 1}`. Bob's system `B` receives `b ∈ {0, 1}`. These settings must be unpredictable by the systems.
    3.  **Outputs (The "Response"):** Each system must produce a binary output `x, y ∈ {-1, +1}` based on the input it received.
    4.  **Data Collection:** We run the experiment for thousands of rounds, covering all four possible input combinations: `(a=0, b=0)`, `(a=0, b=1)`, `(a=1, b=0)`, and `(a=1, b=1)`.
    5.  **Correlation Calculation:** For each of the four settings, we calculate the statistical correlation `E(a, b) = average(x * y)`. This is the average value of the product of the outputs over all rounds with that specific setting.

*   **2.2.2. The Boundaries of Reality**
    The power of the game comes from combining these four correlation values into a single score, `S`, defined by the CHSH inequality:
    `S = | E(0, 0) + E(0, 1) + E(1, 0) - E(1, 1) |`

    The value of `S` tells us which "World" our system inhabits:
    *   **`S ≤ 2` (The Classical Bound):** A fundamental mathematical theorem proven by John Bell and others states that any system that obeys local realism cannot, under any circumstances, produce a score `S` greater than 2.
    *   **`S ≤ 2.828` (The Tsirelson Bound):** Quantum mechanics predicts that systems sharing entangled particles can achieve scores up to `2√2`. This has been repeatedly verified in physical labs.
    *   **`S ≤ 4` (The PR-Box Bound):** The absolute mathematical maximum for any theory that still obeys the Non-Signaling Principle.
    *   **`S > 4` (The Absurdist/Signaling Bound):** Scores greater than 4 are only possible if the system is explicitly "cheating" by sending information between A and B about their input settings.

Our project's entire goal is to train a classical `UniversalController` to drive our classical `ClassicalSubstrate` to achieve scores in the Quantum, Mannequin, and Absurdist regimes, and to map the resources required to do so.

---

### **PART 2: THE UNIVERSAL EXPERIMENTAL FRAMEWORK**


This section defines the core experimental setup. It describes the configurable protocols that allow us to simulate each of the "Four Worlds" and the specific, physically meaningful parameters we will vary to map the computational landscape.

### **3. The Three Primary Experimental Protocols**

The `ExperimentHarness` will be designed as a polymorphic system, capable of running one of three distinct experimental protocols by changing its configuration. This allows for a direct, apples-to-apples comparison of the computational resources required to achieve each level of correlation. The fundamental difference between the protocols lies in the **information and constraints** applied to the `UniversalController`.

The relationship between our theoretical "Four Worlds" model and these implementable protocols is as follows:

| Theoretical World | Governing Principles | Target S-Score | Implemented via |
| :--- | :--- | :--- | :--- |
| **Classical** | Local Realism | `S ≤ 2` | Baseline (no controller or strictly local controller) |
| **Quantum** | Unitary Evolution, Non-Signaling | `S ≤ 2.828` | **Protocol Q** (Physics-informed `QuantumInspiredController`) |
| **Mannequin** | Global Variables, Non-Signaling | `S ≤ 4` | **Protocol M** (Generic `NonLocalCoordinator` with enforced causal separation) |
| **Absurdist** | Explicit Signaling Channel | `S > 4` | **Protocol A** (Controller with a deliberate, measurable communication channel) |


#### **3.1. Protocol Q: The Quantum World Emulator**

*   **Objective:** To achieve a maximum `S`-score that approaches, but does not exceed, the quantum Tsirelson Bound (`S → 2.828`).
*   **Methodology:** This protocol tests the system's ability to emulate known physics. The `UniversalController` is not a generic, black-box learner. Instead, it is implemented as a **`QuantumInspiredController` (QIC)**. Its internal architecture is explicitly designed to perform the correct mathematics of 2-qubit quantum mechanics.
    *   **Internal State:** The QIC maintains its own internal state vector representing the four amplitudes of a 2-qubit system: `[c₀₀, c₀₁, c₁₀, c₁₁]`. This state must be represented using a `torch.complex64` or `torch.complex128` data type.
    *   **Hard-Coded Physics:** The core of the controller's logic is a non-trainable, hard-coded implementation of the **tensor product** and **unitary evolution**. When given the measurement settings `(a, b)`, it constructs the correct `4x4` unitary matrix `U_total = U_A(θ_a) ⊗ U_B(θ_b)` and applies it to its internal state: `ψ_next = U_total @ ψ_current`.
        *   The mapping from CHSH setting bits to measurement angles `θ` must be explicit and configurable, with a default to the optimal CHSH angles (e.g., Alice: {0, π/2}, Bob: {π/4, 3π/4}).
    *   **The Learning Task:** The optimizer's job is **not** to discover quantum mechanics. Its much simpler and more tractable task is to learn the "interface" between the ideal quantum simulation and the messy classical substrate. It trains two small neural networks:
        1.  A "sensor" network that learns the best way to update the internal quantum state based on the observed classical reservoir states.
        2.  An "actuator" network that learns the best way to translate the computed quantum outcomes into effective corrective nudges for the classical reservoirs.
    *   **Optimization Objective:** The primary fitness score for the optimizer is the final `S`-score. However, to guide the search, an auxiliary penalty term should be added to the loss function to penalize deviations of the internal quantum state from unit norm (i.e., `penalty = α * | ||ψ|| - 1 |²`), ensuring the simulation remains physically plausible.
    *   **Measurement:** To produce the binary outcomes required for the CHSH game, the controller must sample from the final quantum state's Born probabilities (`P(outcome) = |amplitude|²`). These outcomes are generated *online* during the trial and are final. This sampling process must use a dedicated, reproducible stream of random numbers.
*   **Significance:** This protocol provides a powerful baseline. If the resulting `S`-score converges to `~2.828`, it proves our framework is capable of perfectly modeling known quantum physics. It allows us to quantify the "cost of reality"—the resources (`R`, `K`) needed for a classical system to perfectly mimic a true quantum system.
*   **Implementation Status:** This protocol is complex and reserved for Track 2. The initial implementation of `UniversalController` should raise a `NotImplementedError` if `protocol='Quantum'` is selected, to prevent accidental and misleading runs.

#### **3.2. Protocol M: The Mannequin World (PR-Box) Emulator**

*   **Objective:** To achieve the maximum possible non-signaling correlation (`S → 4.0`).
*   **Methodology:** This protocol tests the absolute limits of a non-local but non-communicating controller. The `UniversalController` is implemented as a generic **`NonLocalCoordinator` (NLC)** (e.g., a standard Multi-Layer Perceptron).
    *   **No Physics Priors:** The NLC has no built-in knowledge of quantum mechanics. It is a "blank slate" universal function approximator.
    *   **Global View:** It receives the states of both reservoirs (`x_A`, `x_B`) and both input settings (`a`, `b`). This global information access is the "non-local" part.
    *   **The Learning Task:** The optimizer's goal is simply to find the weights for the NLC that maximize the final `S`-score. It is free to discover any mathematical trick or correlation rule that achieves this. The expected emergent strategy is the simple PR-Box rule: `x*y = (-1)^(a AND b)`.
    *   **Non-Signaling Enforcement:** This is the most critical design constraint of Protocol M. A naive implementation that feeds all inputs into a single network and produces both outputs is *implicitly signaling*. To enforce the non-signaling principle, the controller's architecture must be explicitly bifurcated:
        1.  A shared "encoder" network computes a latent representation from the parts of the state that are common knowledge: `latent_L = f_shared(x_A, x_B)`.
        2.  Two separate "decoder" or "head" networks compute the outputs. Each head receives the shared latent state and only its own *local* information:
            *   `output_A = f_head_A(latent_L, x_A, setting_a)`
            *   `output_B = f_head_B(latent_L, x_B, setting_b)`
        This structure makes it architecturally impossible for `setting_b` to influence `output_A`.
    *   **Statistical Audit:** As a secondary check, the `ExperimentHarness` must compute and log empirical non-signaling metrics during evaluation. The metric is the maximum difference in a party's output probability, conditioned on its local setting, as the other party's setting changes.
        *   **Algorithm:** For Alice, `Δ_NS(A) = max_{a,b,b'} | P(y_A=+1|a,b) - P(y_A=+1|a,b') |`. The probabilities are estimated from the frequency of outcomes in the trial data. A 95% confidence interval on this metric should be estimated via bootstrap.
        *   **Enforcement:** A run is considered to have violated the non-signaling constraint if `Δ_NS` exceeds a small threshold (e.g., `ε_default = 0.02`). This can be used as a post-hoc filter or incorporated as a penalty term in the fitness function.
*   **Significance:** This protocol allows us to explore "super-quantum" correlations. It tests the computational power of a system that is freed from the specific mathematical constraints of quantum mechanics but not from causality. The resources (`R`, `K`) required to reach `S=4` can be directly compared to those required to reach `S=2.828` in Protocol Q.

#### **3.3. Protocol A: The Absurdist World (Signaling) Emulator**

*   **Objective:** To demonstrate that `S > 4.0` is achievable and to quantify the relationship between communication bandwidth and the degree of this "super-causal" correlation.
*   **Methodology:** This protocol explicitly and deliberately violates the Non-Signaling principle. It serves as a control experiment to validate our understanding of the system's absolute limits.
    *   **Explicit Signaling Channel:** The `UniversalController` is architected to create a direct, one-way (default: Alice-to-Bob) communication channel. The logic for computing Bob's corrective action `c_B` will receive information derived from Alice's state and/or setting as an explicit input argument.
    *   **The Learning Task:** The optimizer will now learn to exploit this direct channel. This is instrumented by having the controller learn to encode information into a latent vector, which is then quantized to a specified bit budget (`b_signal`) before being passed to the other party.
        *   **Quantization Primitive:** A compliant implementation must support a `quantize(vector, bits, mode)` function that reduces a real-valued vector to a `bits`-length bitstring. The empirical Shannon entropy of the transmitted bitstrings must be logged and shown to be ≤ `bits`.
    *   **The Independent Variable:** In this protocol, we will introduce a new "knob": the **bandwidth of the signaling channel** (`b_signal`). We will constrain the information passed from Alice's side to Bob's to a single bit, 2 bits, 4 bits, etc.
*   **Significance:** This protocol is not intended to model a plausible physical reality. Its purpose is to act as a **diagnostic tool**. It allows us to create a plot of `S_max` vs. "Bits of Signaling," which provides a concrete, information-theoretic grounding for the entire framework. A sanity check for this protocol is to compare the empirical results to the analytic expectation for `S` given a simple signaling strategy. It demonstrates that we understand the system's behavior so well that we can make it break any physical bound by a controllable amount.
*   **Implementation Status:** Protocol A will be the focus of Track 3. Until then, selecting `protocol='Absurdist'` may raise a `NotImplementedError` or run in a simplified diagnostic mode depending on the configuration.

#### 3.4. Normative Information-Flow Contract

Any compliant implementation, regardless of hardware or software framework, MUST adhere to the following data dependency constraints for each protocol. A violation of this table constitutes a failed implementation of the protocol.

| Signal Path                | Classical Baseline | **Protocol Q**                                                        | **Protocol M**                                       | **Protocol A (A→B signal)**                           |
| -------------------------- | ------------------ | --------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| **A local setting → A output** | Allowed            | Allowed                                                               | Allowed                                              | Allowed                                               |
| **B local setting → A output** | **Forbidden**      | **Forbidden**                                                         | **Forbidden**                                        | **Forbidden**                                         |
| **A state → B output**         | Forbidden          | Indirectly, via updates to shared quantum state `ψ`                   | Allowed via **shared encoder only** (no local setting data) | Allowed via **quantized, bit-bounded signaling channel** |
| **A setting → B output**       | **Forbidden**      | Indirectly, via updates to shared quantum state `ψ`                   | **Forbidden**                                        | Allowed via **quantized, bit-bounded signaling channel** |


---

### **4. The Universal Control Parameters**

To make our results generalizable and comparable across different physical systems, we define our experimental "knobs" as dimensionless parameters.

#### **4.1. Speed Ratio (`R`)**
*   **Definition:** `R = τ_slow / τ_fast`, where `τ_slow` is the characteristic update time of the substrate and `τ_fast` is the end-to-end latency of the controller.
*   **Significance:** This parameter quantifies the fundamental "superpower" of the layered-time architecture. It measures how many computational "thoughts" the controller can have for every single "moment" the substrate experiences.
*   **Implementation:**
    > **Normative:** An implementation MUST log both the configured delay `d_config` *and* the *effective speed ratio* `R_effective`.
    > *   When `d_config ≥ 1`: `R_effective = 1 / d_config`.
    > *   When `0 < d_config < 1`: `n_internal = round(1/d_config)`, and `R_effective = n_internal`.
    >
    > **Canonical Timing Mapping:** Let the substrate's characteristic time `τ_slow = 1` (a single `step()` of the ESN). Let `d` be the controller delay in slow ticks. The speed ratio `R` is then defined as `R = 1/d`.
    > *   **Lagged Regime (`d ≥ 1`):** The controller is slower than the substrate (`R ≤ 1`). The corrective signal `c(k)` computed from state `x(k)` is applied at a future step `k+d`. This is managed by a simple delay buffer.
    > *   **Accelerated Regime (`0 < d < 1`):** The controller is faster than the substrate (`R > 1`). The controller performs `n = round(1/d)` internal update computations based *only* on the state `x(k)`. This is implemented with a recurrent internal hidden state in the controller (e.g., a `nn.GRUCell`), which is updated for `n` steps before a final correction is emitted. This robustly models "more thinking time" on a static input. This feature can be enabled/disabled via a configuration flag for ablation studies.
*   **Target Test Set:** The primary `S(R)` curve will be generated by sweeping `d` across a logarithmic-like scale, e.g., `{0.25, 0.5, 1, 2, 4, 8, 16}`.

#### **4.2. Information Ratio (`K`)**
*   **Definition:** `K = I_controller / I_substrate`, where `I` represents the information capacity of a component.
*   **Significance:** This parameter quantifies the "intelligence" or "complexity" of the controller relative to the system it is controlling. The results of our `goldilocks_sweep` suggest that the relationship between `K` and performance is highly non-monotonic, with a distinct "Goldilocks Zone" of optimal complexity.
*   **Implementation:**
    *   `I_substrate`: The number of state variables, `N_A + N_B`.
    *   `I_controller`: The number of trainable weights and biases in the `UniversalController`'s neural network. For rigorous analysis, we should report both the raw parameter count and an estimate of the true information content (e.g., 32 bits per float parameter).
*   **Target Test Set:** We will generate multiple `S(R)` curves, each for a different value of `K`. This will be done by running the experiment with different controller architectures (e.g., hidden layers of size {16, 32, 64, 128}). This will produce a 3D surface plot, `S(R, K)`, as the main deliverable of Track 1.

#### **4.3. Sensor Noise (`σ_noise`)**
*   **Definition:** The standard deviation of zero-mean Gaussian noise that is added to the substrate's state vector *before* it is passed to the controller.
    `input_to_controller = x_substrate + N(0, σ_noise)`
*   **Significance:** This parameter models the real-world imperfection of measurement devices. It allows us to test the robustness of our learned control policies. A truly powerful controller should be able to function even with noisy, incomplete information.
*   **Implementation:** This will be a configurable parameter in the `ExperimentHarness`. The noise is added to the substrate's state vector *after* it has been retrieved for the controller, ensuring it does not affect the substrate's internal dynamics, only the controller's "view." We will run dedicated experiments (e.g., as part of the robustness checks in the final phase) to plot `S_max` as a function of `σ_noise` for a fixed, high-performing `(R, K)` configuration.

#### **4.4. Measurement and Scoring Protocol**

To ensure that the final `S`-score is well-defined and reproducible, the process of extracting binary outcomes from the continuous system dynamics and calculating the score must be rigidly specified.

*   **Online Output Generation:** During each trial, each protocol emits a scalar logit per party. The harness maps this to a binary CHSH outcome `{-1, +1}` immediately using the configured `binarization_mode` (default: `sign(logit)`). These per-trial binary values are *final* and are what enter the CHSH tallies. It is critical to note that the same scalar output from the controller is used for both the corrective action (actuation) and the CHSH outcome (measurement logit), though an `actuation_scale` parameter can be used to decouple their magnitudes. Post-hoc regression/readout is permitted only for *diagnostic analysis* and must not alter the recorded outcomes.
*   **Balanced Datasets & Washout:** The CHSH game requires averaging over many trials with balanced settings.
    *   The file containing the input settings `(a,b)` for each trial must be generated in advance with an equal, block-shuffled number of instances for each of the four setting combinations.
    *   If a `washout_steps` period is used, its value must be a multiple of 4 to ensure the remaining trials for scoring stay balanced.
    *   The total number of simulation steps, `T_total`, should be equal to the number of settings provided. The harness must `assert len(chsh_settings) == T_total`.
    *   A final sanity check after the loop should confirm that the counts of each setting in the scored data (`settings_log`) are equal, warning the user of any accidental imbalance.
*   **Statistical Significance:** A single `S`-score is not sufficient. To claim a violation of a classical or quantum bound, the statistical significance must be reported.
    *   **Method:** The recommended method is to compute a p-value via a **one-sided** bootstrap test using a dedicated, seeded RNG for reproducibility. After a run is complete, the set of all trial outcomes is resampled with replacement (e.g., `10,000` times) to generate a distribution of `S`-scores. The p-value is the fraction of these resampled scores that fall at or below the bound being tested (e.g., `S_null = 2.0` for the classical bound). This utility, along with the CHSH scorer and non-signaling metric, should be centralized (e.g., in `apsu6/metrics.py`).

---


### **PART 3: DETAILED COMPONENT SPECIFICATIONS & IMPLEMENTATION**


This section provides reference implementations using Python-style pseudocode. These are illustrative and non-normative; any compliant implementation may use different frameworks (e.g., JAX, FPGA HDL) provided it honors the Normative Interface Contracts defined below.

#### 5.0.1. Normative Interface Contract

*   **State Types:** Substrate state vectors `x_A`, `x_B` are 1-D real-valued arrays of length `N_A`, `N_B` respectively. Precision must be equivalent to or greater than 16-bit float.
*   **Settings Domain:** The raw CHSH settings `a, b` are bits `{0, 1}`. A *controller-internal* mapping to spin domain `ζ = 2*bit - 1 ∈ {-1, +1}` is used for computation. The substrate itself always receives raw bits as part of its input vector.
*   **Corrections Domain:** The controller produces a real-valued scalar logit per party. The sign of this logit determines the CHSH outcome `{-1, +1}`. The magnitude, scaled by `actuation_scale`, is used for the substrate correction signal.
*   **Batching:** The reference implementation assumes a batch size of 1 for each time step. Compliant implementations MAY vectorize for performance but MUST preserve the temporal order of operations and trial independence.
*   **Reset Semantics:** A `reset()` call on any component must return it to a state that is statistically independent of any prior trials (e.g., zero state or re-initialization from a new seed).
*   **Time Update:** One `step()` of the substrate consumes exactly one input vector per party.
*   **Randomness:** All stochastic operations (noise, sampling, initialization) MUST be fully determined by the seeded RNG streams defined in the registry in §9.2.1.
*   **Parameter Loading:** Implementations MUST provide a deterministic function to inject/load all trainable controller parameters prior to an evaluation run.

### **5. The `ClassicalSubstrate` Module**

*   **Filename:** `apsu6/substrate.py`
*   **Purpose:** This module represents the physical "slow medium" being controlled. It encapsulates the classical dynamical systems, their state, and their response to external inputs. Its design must ensure that the two subsystems (A and B) are computationally independent and that their state can be reliably reset and diagnosed.

#### **5.1. Class Definition: `ClassicalSubstrate`**

```python
# --- Reference Python-style pseudocode (non-normative) ---
# Any compliant implementation MAY substitute its own tensor/array types
# provided the interface contracts are honored.
#
# Pseudocode placeholders:
#   TensorLib = user-chosen dense tensor library (e.g., PyTorch, JAX, NumPy)
#   ReservoirLib = user-chosen ESN / dynamical substrate backend

class ClassicalSubstrate:
    """
    Encapsulates the two Echo State Network (ESN) reservoirs representing the
    "slow medium" of the experiment (Systems A and B). This implementation
    uses the 'reservoirpy' library.
    """
    def __init__(self, N_A, N_B, sr_A, sr_B, lr_A, lr_B, noise_A, noise_B, seed_A, seed_B, device):
        """
        Initializes two distinct ESN reservoirs.

        Args:
            N_A (int): Number of units in reservoir A.
            N_B (int): Number of units in reservoir B.
            sr_A (float): Spectral radius for reservoir A.
            ...and so on for all hyperparameters.
            seed_A (int): Random seed for reservoir A's internal matrices.
            seed_B (int): Random seed for reservoir B's internal matrices.
            device (torch.device): The compute device ('cpu' or 'cuda'). Note: reservoirpy
                                 is primarily NumPy/CPU-based. For high-throughput GPU
                                 scenarios, data must be moved between devices each step,
                                 which may incur performance overhead. A future version might
                                 consider a pure-torch library like EchoTorch.
        """
        self.device = device
        self.N_A, self.N_B = N_A, N_B

        # The input to the substrate is a 2D vector [setting, correction]
        input_dim = 2
        # Rationale: Using distinct seeds ensures there is no accidental correlation
        # in their internal random wiring.
        self.reservoir_A = rpy.nodes.Reservoir(units=N_A, input_dim=input_dim, sr=sr_A, lr=lr_A, noise_rc=noise_A, seed=seed_A)
        self.reservoir_B = rpy.nodes.Reservoir(units=N_B, input_dim=input_dim, sr=sr_B, lr=lr_B, noise_rc=noise_B, seed=seed_B)
        
        # State is managed internally by reservoirpy nodes.
        self.reset()

    def step(self, input_A, input_B):
        """
        Evolves the system by one time step. Inputs are expected to be NumPy arrays.

        Args:
            input_A (np.ndarray): The input signal for reservoir A, shape (1, 2).
            input_B (np.ndarray): The input signal for reservoir B, shape (1, 2).

        Returns:
            (np.ndarray, np.ndarray): The new state vectors x_A(t+1), x_B(t+1).
        """
        # .run() is stateful in reservoirpy. It updates the internal state.
        state_A = self.reservoir_A.run(input_A)
        state_B = self.reservoir_B.run(input_B)
        return state_A, state_B

    def get_state(self):
        """
        Returns the current internal states of the reservoirs as detached torch tensors.
        """
        state_A_np = self.reservoir_A.state()
        state_B_np = self.reservoir_B.state()
        # Ensure conversion to torch tensors for the controller
        state_A = torch.from_numpy(state_A_np).float().to(self.device)
        state_B = torch.from_numpy(state_B_np).float().to(self.device)
        # Ensure state has a batch dimension for the controller
        if state_A.ndim == 1: state_A = state_A.unsqueeze(0)
        if state_B.ndim == 1: state_B = state_B.unsqueeze(0)
        return state_A, state_B

    def reset(self):
        """
        Resets the internal state of both reservoirs to zero.
        This is a CRITICAL method for ensuring trial independence.
        """
        # reservoirpy's reset method handles this.
        self.reservoir_A.reset()
        self.reservoir_B.reset()

    def diagnose(self):
        """
        Performs the pre-flight check as per spec, generating plots for both
        reservoirs to ensure they are healthy and computationally rich.
        """
        # ... (Implementation as detailed in v3.0 document) ...
        print("Running diagnostics for Reservoir A...")
        # ... generate plots for A ...
        print("Running diagnostics for Reservoir B...")
        # ... generate plots for B ...
```

---

### **6. The `UniversalController` Module**

*   **Filename:** `apsu6/controller.py`
*   **Purpose:** This is the heart of the "fast medium." It is a polymorphic class designed to be configurable to run any of the three experimental protocols (Quantum, Mannequin, Absurdist). It takes the state of the substrate as input and produces corrective actions.

#### **6.1. Class Definition: `UniversalController`**

```python
# --- Reference Python-style pseudocode (non-normative) ---
#
# Pseudocode placeholders:
#   TensorLib = user-chosen dense tensor library (e.g., PyTorch, JAX, NumPy)

class UniversalController(TensorLib.Module):
    """
    A polymorphic controller that can be configured to emulate different
    physical realities (Quantum, Mannequin, Absurdist).
    """
    def __init__(self, protocol, N_A, N_B, K_controller, R_speed, signaling_bits, internal_cell_config, device):
        """
        Initializes the controller for a specific protocol.

        Args:
            protocol (str): 'Quantum', 'Mannequin', or 'Absurdist'.
            N_A (int): The dimension of substrate A's state space.
            N_B (int): The dimension of substrate B's state space.
            K_controller (int): A parameter controlling the controller's complexity (e.g., hidden layer size).
            R_speed (float): The speed ratio, used to configure internal loop iterations for d < 1.
            signaling_bits (int): For Protocol A, the bandwidth of the signaling channel.
            internal_cell_config (dict): Configuration for the internal recurrent cell for R>1 mode.
            device (torch.device): The compute device for torch tensors.
        """
        super().__init__()
        self.protocol = protocol
        self.device = device
        self.R = R_speed
        self.signaling_bits = signaling_bits
        self.internal_cell_config = internal_cell_config

        # Build the neural network architecture based on the protocol
        self._build_network(N_A, N_B, K_controller)
        
        # Build internal cell for R>1 "thinking" if enabled
        if self.R > 1 and self.internal_cell_config.get('enabled', False):
            cell_type = self.internal_cell_config['type']
            hidden_size = self.internal_cell_config['hidden_size']
            
            # Input to the shared "thinking" cell must not leak local settings
            # for non-signaling protocols.
            if self.protocol == 'Mannequin':
                cell_input_size = N_A + N_B
            else: # Other protocols can include settings if appropriate
                cell_input_size = N_A + N_B + 2
            
            if cell_type == 'gru':
                self.internal_cell = nn.GRUCell(cell_input_size, hidden_size).to(device)
            # The decoder maps the "thought" vector back to the controller's main pathway.
            # We map back to K_controller to be used by the protocol-specific heads.
            self.internal_decoder = nn.Linear(hidden_size, K_controller).to(device)

        self.to(self.device)

    def _build_network(self, N_A, N_B, K_controller):
        """Helper method to construct the appropriate NN architecture."""
        N_substrate = N_A + N_B
        if self.protocol == 'Quantum':
            # Protocol Q: A structured network with hard-coded physics
            # Implementation is deferred to Track 2.
            self.quantum_state_dim = 4 # Still define attr for reset() guard
            pass
        elif self.protocol == 'Mannequin':
            # Protocol M: Enforces non-signaling via architecture.
            self.shared_encoder = nn.Sequential(
                nn.Linear(N_substrate, K_controller),
                nn.ReLU()
            )
            self.head_A = nn.Sequential(
                nn.Linear(K_controller + N_A + 1, K_controller), # latent + x_a + setting_a
                nn.ReLU(),
                nn.Linear(K_controller, 1) # output c_a
            )
            self.head_B = nn.Sequential(
                nn.Linear(K_controller + N_B + 1, K_controller), # latent + x_b + setting_b
                nn.ReLU(),
                nn.Linear(K_controller, 1) # output c_b
            )

        elif self.protocol == 'Absurdist':
            # Protocol A: A generic MLP whose input size will be expanded by the
            # harness to include the signaling channel.
            mlp_input_size = N_A + N_B + 2 + self.signaling_bits # x_a, x_b, setting_a, setting_b, signal_vec
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_size, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, 2), # Output for c_A and c_B
                nn.Tanh() # Bounded output for stability
            )
    
    def forward(
        self,
        x_A: torch.Tensor,         # Shape: (B, N_A)
        x_B: torch.Tensor,         # Shape: (B, N_B)
        settings_A: torch.Tensor,  # Shape: (B, 1), spin domain {-1, +1}
        settings_B: torch.Tensor   # Shape: (B, 1), spin domain {-1, +1}
    ) -> torch.Tensor:             # Shape: (B, 2), logits
        """The main forward pass, dispatched based on protocol."""
        
        # --- R>1: Internal "thinking" loop ---
        internal_iterations = max(1, int(round(self.R))) if self.R > 1 else 1
        if internal_iterations > 1 and hasattr(self, 'internal_cell'):
            # This logic block simulates "more thinking time" on a static input
            # Crucially, the input to this shared cell must not violate protocol constraints.
            if self.protocol == 'Mannequin':
                 cell_input = torch.cat([x_A, x_B], dim=-1)
            else:
                 cell_input = torch.cat([x_A, x_B, settings_A, settings_B], dim=-1)

            hidden_state = torch.zeros(x_A.size(0), self.internal_cell.hidden_size, device=self.device)
            for _ in range(internal_iterations):
                hidden_state = self.internal_cell(cell_input, hidden_state)
            # The final "thought" is decoded and used as a prefix for protocol logic
            thought_vector = self.internal_decoder(hidden_state)
        else:
            thought_vector = None # No "thinking" performed

        # --- Protocol-specific logic ---
        correction_A, correction_B = None, None # Ensure assignment in all branches
        if self.protocol == 'Quantum':
            raise NotImplementedError("Protocol Q is not yet implemented.")
            
        elif self.protocol == 'Mannequin':
            # Use thought_vector if available, otherwise use raw substrate state
            prefix = thought_vector if thought_vector is not None else torch.cat([x_A, x_B], dim=-1)
            # The shared_encoder input size must match the prefix size
            shared_latent = self.shared_encoder(prefix)
            
            # --- Non-Signaling architectural assertion (conceptual) ---
            # assert 'setting_B' not in head_A_inputs and 'setting_A' not in head_B_inputs
            input_A = torch.cat([shared_latent, x_A, settings_A], dim=-1)
            correction_A = self.head_A(input_A)

            input_B = torch.cat([shared_latent, x_B, settings_B], dim=-1)
            correction_B = self.head_B(input_B)
            
        elif self.protocol == 'Absurdist':
            # The harness is responsible for creating and passing the signal vector.
            # This is a placeholder for a learned, quantized encoding.
            signal_vec = self._encode_signal(x_A, settings_A)
            controller_input = torch.cat([x_A, x_B, settings_A, settings_B, signal_vec], dim=-1)
            
            corrections = self.mlp(controller_input) 
            correction_A, correction_B = corrections[..., 0:1], corrections[..., 1:2]

        # Standardize return shape to (batch, 2)
        return torch.cat([correction_A, correction_B], dim=-1)

    def _encode_signal(self, x, s):
        """Placeholder for a learned, deterministic, bandwidth-limited signal encoder."""
        # A real implementation would involve a learned MLP and a quantization layer.
        # This placeholder is for shape compatibility only.
        return torch.zeros(x.size(0), self.signaling_bits, device=self.device)

    def reset(self):
        """Resets any internal state of the controller."""
        if self.protocol == 'Quantum':
            raise NotImplementedError("Protocol Q reset is not yet implemented.")
            # Reset to the initial Bell state
            # self.internal_quantum_state = torch.zeros(...)
```

*   **Defensive Design:** The class is explicitly polymorphic. The choice of which "universe" to simulate is a high-level configuration parameter, preventing accidental mixing of protocols. The use of separate network heads (`sensor_head`, `actuator_head`) for Protocol Q makes the learned components modular and analyzable. The `Tanh` output on the MLP provides crucial stability.

---

### **7. The `ExperimentHarness` Module**

*   **Filename:** `apsu6/harness.py`
*   **Purpose:** This is the main orchestration script. It is the "lab bench" that sets up the experiment, runs the simulation loop, gathers data, and computes the final fitness score.

#### **7.1. Class Definition: `ExperimentHarness`**

```python
# --- Reference Python-style pseudocode (non-normative) ---
#
# Pseudocode placeholders:
#   TensorLib = user-chosen dense tensor library (e.g., PyTorch, JAX, NumPy)
#   ArrayLib = user-chosen array library (e.g., NumPy)

# Example imports for a concrete implementation:
# from collections import Counter
# from .substrate import ClassicalSubstrate
# from .controller import UniversalController
# from .metrics import calculate_chsh_score, calculate_nonsignaling_metric
# from .utils import load_randomness, bits_to_spins

class DelayBuffer:
    """
    Handles the delay `d` for the controller's correction signals.
    """
    def __init__(self, delay, num_channels=2, device='cpu', dtype=torch.float32):
        """If delay is 0, the buffer acts as a passthrough."""
        self.delay = int(delay)
        if self.delay < 0:
            raise ValueError("Delay cannot be negative.")
        self.num_channels = num_channels
        self.buffer = torch.zeros(self.delay, self.num_channels, device=device, dtype=dtype)
    
    def push(self, signal: torch.Tensor):
        """Pushes a new signal in, returns the oldest one."""
        # Assert that the harness is running in batch=1 mode for the buffer.
        assert signal.numel() == self.num_channels or signal.shape[0] == 1
        signal_1d = signal.detach().squeeze().view(self.num_channels)
        if self.delay == 0:
            return signal_1d
        
        oldest_signal = self.buffer[0, :].clone()
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        self.buffer[-1, :] = signal_1d
        return oldest_signal
        
    def reset(self):
        self.buffer.zero_()

class ExperimentHarness:
    def __init__(self, config):
        """
        Initializes the entire experiment from a configuration dictionary.
        
        Args:
            config (dict): A dictionary specifying all parameters for the
                           substrate, controller, protocol, and optimization.
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.substrate = ClassicalSubstrate(**config['substrate_params'], device=self.device)
        self.controller = UniversalController(**config['controller_params'], device=self.device)
        
        # Setup delay buffer for R < 1 regime
        controller_delay = config.get('controller_delay', 1)
        # For R > 1 (d < 1), controller iterates internally, no external delay needed.
        self.delay_buffer = DelayBuffer(delay=controller_delay if controller_delay >= 1 else 0, device=self.device)
        
        # Load pre-generated, cryptographically secure randomness
        self.chsh_settings = load_randomness(config.get('randomness_file'))

        # Setup dedicated RNG for sensor noise for reproducibility
        self.noise_rng = torch.Generator(device=self.device)
        self.noise_rng.manual_seed(config.get('noise_seed', 42))

 
    def _build_input(self, setting_bit: float, correction_val: float) -> np.ndarray:
        """
        Composes the substrate drive vector for one party.
        The setting and correction are concatenated, not added.

        Returns:
            np.ndarray: Shape (1, 2): [setting_bit, correction_val].
        """
        return np.array([[setting_bit, correction_val]], dtype=np.float32)
 
    def evaluate_fitness(self, controller_weights):
        """
        Performs one full fitness evaluation for a given set of controller weights.
        This is the function that the GlobalOptimizer will call repeatedly.
        The optimizer wrapper must handle unpacking the returned tuple.
        """
        # 1. SETUP
        self.substrate.reset()
        self.controller.reset()
        self.delay_buffer.reset()
        self.controller.load_state_dict(controller_weights) # Assumes torch state_dict
        self.controller.eval()
        
        # Introduce sensor noise if specified in config
        sensor_noise_std = self.config.get('sensor_noise', 0.0)
        actuation_scale = self.config.get('actuation_scale', 1.0)

        # 2. SIMULATION LOOP
        outputs_A, outputs_B, settings_log = [], [], []
        T_total = self.config.get('T_total', 4000)
        assert len(self.chsh_settings) == T_total, "CHSH settings file length must match T_total."

        washout_steps = self.config.get('washout_steps', 100)
        assert washout_steps % 4 == 0, "Washout steps must be a multiple of 4 to maintain CHSH balance."

        with torch.no_grad():
            for t in range(T_total):
                # 2a. Get current state and apply sensor noise for controller
                state_A, state_B = self.substrate.get_state() # clean state, shape (B, N)
                
                noisy_state_A = state_A + torch.randn_like(state_A, generator=self.noise_rng) * sensor_noise_std
                noisy_state_B = state_B + torch.randn_like(state_B, generator=self.noise_rng) * sensor_noise_std
                
                # 2b. Get CHSH settings for this time step
                setting_A_bit, setting_B_bit = self.chsh_settings[t]
                
                # Convert settings to spin domain {-1, +1} for the controller
                setting_A_spin, setting_B_spin = bits_to_spins((setting_A_bit, setting_B_bit))
                setting_A_tensor = torch.tensor([[setting_A_spin]], device=self.device)
                setting_B_tensor = torch.tensor([[setting_B_spin]], device=self.device)
                 
                # 2c. Compute correction logits from controller
                correction_logits = self.controller.forward(noisy_state_A, noisy_state_B, setting_A_tensor, setting_B_tensor)
                logit_A, logit_B = correction_logits[..., 0], correction_logits[..., 1]

                # 2d. Apply delay `d` via buffer
                delayed_logits = self.delay_buffer.push(correction_logits)
                delayed_logit_A, delayed_logit_B = delayed_logits[0], delayed_logits[1]

                # 2e. Evolve substrate using scaled correction and raw bit setting
                substrate_input_A = self._build_input(float(setting_A_bit), delayed_logit_A.item() * actuation_scale)
                substrate_input_B = self._build_input(float(setting_B_bit), delayed_logit_B.item() * actuation_scale)
                self.substrate.step(substrate_input_A, substrate_input_B)

                # 2f. Record final outputs for this trial (online binarization)
                if t >= washout_steps:
                    binarization_mode = self.config.get('binarization_mode', 'sign')
                    if binarization_mode == 'sign':
                        y_A = 1 if logit_A.item() >= 0 else -1
                        y_B = 1 if logit_B.item() >= 0 else -1
                    else: # 'bernoulli', etc.
                        raise NotImplementedError(f"Binarization mode '{binarization_mode}' not implemented.")
                    outputs_A.append(y_A)
                    outputs_B.append(y_B)
                    settings_log.append((setting_A_bit, setting_B_bit))

        # Sanity check that setting counts are balanced after washout
        counts = Counter(settings_log)
        if len(counts) == 4 and not all(c == len(settings_log)//4 for c in counts.values()):
            print(f"Warning: CHSH setting counts are unbalanced after washout: {counts}")

        # 3. SCORING
        S_score, correlations = calculate_chsh_score(outputs_A, outputs_B, settings_log)
        
        # 4. DIAGNOSTICS
        diagnostics = self._compute_diagnostics(S_score, correlations, outputs_A, outputs_B, settings_log)
        
        return S_score, diagnostics # Return primary fitness and rich data packet

    def _compute_diagnostics(self, s_score, correlations, outputs_A, outputs_B, settings_log):
        """Computes and returns a dictionary of diagnostic metrics."""
        settings_a_bits = [s[0] for s in settings_log]
        settings_b_bits = [s[1] for s in settings_log]

        delta_ns_A, ci_A = calculate_nonsignaling_metric(outputs_A, settings_a_bits, settings_b_bits)
        delta_ns_B, ci_B = calculate_nonsignaling_metric(outputs_B, settings_b_bits, settings_a_bits)
        
        balance_ok = all(c == len(settings_log)//4 for c in Counter(settings_log).values())

        R_cfg = self.config.get('controller_delay', 1.0)
        R_effective = 1.0 / R_cfg if R_cfg >= 1 else round(1.0 / R_cfg)

        # Controller param counting helper (backend-specific)
        # I_controller_params = count_trainable_params(self.controller)
        I_controller_params = sum(p.numel() for p in self.controller.parameters())
        I_substrate = self.config['substrate_params']['N_A'] + self.config['substrate_params']['N_B']
        K_effective = I_controller_params / I_substrate if I_substrate > 0 else 0

        return {
           "S_score": s_score,
           "correlations": correlations,
           "non_signaling_metric_A": delta_ns_A,
           "non_signaling_metric_A_ci": ci_A,
           "non_signaling_metric_B": delta_ns_B,
           "non_signaling_metric_B_ci": ci_B,
           "chsh_balance_ok": balance_ok,
           "R_effective": R_effective,
           "K_effective": K_effective,
           "I_controller_params": I_controller_params,
           "I_substrate": I_substrate,
           "config": self.config,
           # ... other metrics like FLOPs, wall_time, etc.
        }
```

*   **Defensive Design:**
    *   **Configuration-Driven:** The entire experiment is driven by a single configuration dictionary/file. This ensures that every parameter is explicitly defined and logged, preventing "magic numbers" and enhancing reproducibility.
    *   **Decoupling:** The harness cleanly separates the substrate, controller, and evaluation logic. This modularity allows us to swap out components (e.g., use a different reservoir model or a different optimizer) with minimal code changes.
    *   **Statelessness:** The `evaluate_fitness` function is designed to be fully self-contained. It re-initializes and resets everything at the start, guaranteeing that each trial is independent and free from state leakage.

---

### **PART 4: THE RESEARCH PROGRAM & PROTOCOLS**


This final section details the strategic execution of the project. It organizes the experimental work into a series of parallel research tracks, each designed to answer a specific scientific question. It also defines the rigorous protocols for data management and publication that will ensure the credibility and impact of our findings.

### **8. The Research Program: A Multi-Track Investigation**

The project will proceed along five primary research tracks. Tracks 0 and 4 are foundational, providing tools and insights for the main experimental tracks (1, 2, and 3). This parallel structure allows for flexibility and maximizes our rate of discovery.

#### **8.1. Track 0: Meta-Optimization (The Search for an Optimal Optimizer `Φ`)**

*   **Goal:** To identify the most computationally efficient search algorithm for the specific fitness landscapes encountered in this project, thereby accelerating all other research tracks.
*   **Scientific Rationale:** The choice of optimizer is not neutral; different algorithms have different inductive biases and excel on different types of problems. A "brute-force" approach with a generic optimizer may be computationally wasteful or fail to find deep optima. This track aims to find the "right tool for the job."
*   **Methodology (Phase 0a: Optimizer Bake-Off):**
    1.  **Benchmark Problem:** A single, fixed experimental configuration will be used as the testbed (e.g., Protocol M, `K` corresponding to a 32-unit NLC, `R` corresponding to `d=1`).
    2.  **Contenders:** A suite of four diverse, state-of-the-art, gradient-free optimizers will be benchmarked:
        *   **CMA-ES (Evolutionary Strategy):** Our current baseline.
        *   **Differential Evolution (DE):** A population-based algorithm known for its strength on multi-modal landscapes.
        *   **Bayesian Optimization (BO):** A sample-efficient, model-based approach ideal for expensive fitness functions.
        *   **Simulated Annealing (SA):** A trajectory-based method that will provide insight into the "funnel-like" nature of the landscape.
    3.  **Metric:** The primary metric will be **"Time-to-Threshold,"** defined as the number of fitness function evaluations required for each optimizer to reliably achieve a target `S`-score (e.g., `S=2.1`).
*   **Deliverable:** A recommendation for the default optimization algorithm to be used in all subsequent tracks, along with a report on the comparative performance of each contender.

#### **8.2. Track 1: The `S(R, K)` Surface (Mapping the Mannequin World)**

*   **Goal:** To generate the primary scientific result of the project: a quantitative map of the relationship between controller resources and achievable non-local correlation.
*   **Methodology:**
    1.  **Protocol:** This track will exclusively use **Protocol M (The Mannequin World Emulator)**. The `UniversalController` will be a generic MLP, free to discover the optimal correlation strategy while respecting the non-signaling architecture.
    2.  **Grid Search:** A grid search will be performed over the two primary independent variables:
        *   **Speed Ratio (`R`):** Sweeping the delay parameter `d` across `{0.25, 0.5, 1, 2, 4, 8, 16}`.
        *   **Information Ratio (`K`):** Sweeping the controller complexity by using NLCs with hidden layer sizes of `{8, 16, 32, 64, 128}`.
    3.  **Optimization:** For each `(R, K)` point on the grid, the `GlobalOptimizer` (selected from Track 0) will be run for a full, long-duration optimization (e.g., 1000 generations or a fixed wall-clock budget) to find the maximum `S`-score. At least 5 independent replicates per grid point should be run to establish confidence intervals.
        *   **Seeding Strategy:** Replicates will be seeded deterministically using a base seed and the replicate index, e.g., `final_seed = seed_base + replicate_id`. Both values must be logged.
*   **Deliverable:**
    1.  A 3D surface plot of `S_max(R, K)`, visualizing the trade-offs between controller speed and complexity.
    2.  A 2D contour plot identifying the "Goldilocks Zone"—the region of `(R, K)` space that yields the highest performance. This plot will be the central figure of our first major publication.
    3.  Analysis of derived metrics such as `S(R*K)` or `S(FLOPs)` to search for underlying scaling laws.

#### **8.3. Track 2: The Limits of Quantum Emulation (Executing Protocol Q)**

*   **Goal:** To determine if our framework can successfully reproduce the precise statistical limits of known quantum physics (i.e., the Tsirelson Bound).
*   **Methodology:**
    1.  **Protocol:** This track will exclusively use **Protocol Q (The Quantum World Emulator)**. The `UniversalController` will be the physics-informed `QuantumInspiredController` with hard-coded quantum mathematics.
    2.  **Optimization:** The optimizer will find the optimal parameters for the "sensor" and "actuator" sub-networks of the QIC. The `(R, K)` configuration will be informed by the results of Track 1, but may require fine-tuning, while being mindful of the risk of compounding noise from the previous search.
*   **Deliverable:** A final, best-achieved `S`-score from the QIC, with tight confidence intervals derived from multiple runs. A successful result would be `S` converging precisely to `2.828 ± ε`, demonstrating a perfect emulation.

#### **8.4. Track 3: The Cost of Signaling (Executing Protocol A)**

*   **Goal:** To validate our understanding of the `S=4` boundary and to quantify the relationship between information bandwidth and the degree of causal violation.
*   **Methodology:**
    1.  **Protocol:** This track will exclusively use **Protocol A (The Absurdist World Emulator)**, which features an explicit signaling channel.
    2.  **Bandwidth as a Variable:** We will treat the information capacity of the signaling channel as our independent variable. This will be implemented by quantizing the signal passed from the Alice-side to the Bob-side of the controller to `b` bits, where `b` is swept across `{1, 2, 4, 8, 16, 32}`.
*   **Deliverable:** A plot of `S_max` vs. Signaling Bandwidth (`b`). This will quantitatively demonstrate how much "cheating" is required to achieve "impossible" `S > 4` scores.

#### **8.5. Track 4: Universal Reservoir Characterization (The "Natural Systems" Interface)**

*   **Goal:** To develop a practical methodology for harnessing computation in arbitrary, "found" physical systems, bridging the gap from our simulation to real-world hardware.
*   **Methodology:** This track is a self-contained engineering project.
    1.  **Phase A: The "Reservoir Fingerprint" Protocol.** Develop a standardized software suite, `apsu-fingerprint`, that takes a time-series recording from any black-box dynamical system and outputs a vector of its key computational properties (Memory Capacity, Non-Linearity, etc.).
    2.  **Phase B: The "Ad-Hoc Controller" Training.** Develop a method to use this "fingerprint" to automatically configure an optimal `W_in` and `W_out` mapping for a given task, allowing for the rapid, ad-hoc use of any complex system as a computational reservoir.
*   **Deliverable:** An open-source Python package, `apsu-fingerprint`, and a technical report demonstrating its use on both simulated and real-world data (e.g., weather data, EEG signals).

---

### **9. Data Management, Reproducibility, and Publication Strategy**

The credibility of this project rests on unimpeachable rigor and transparency.

#### **9.1. Immutable Experiment Tracking & Version Pinning**
*   **Run Manifest:** Every experiment run will be assigned a unique identifier (UUID) and generate a comprehensive manifest file (e.g., `manifest.json`). All artifacts (logs, plots, saved models, configurations) will be stored in a directory named with this UUID. The manifest must contain:
    *   **Configuration:** The full experimental config (protocol, R, K, σ_noise, etc.).
    *   **Versioning:** The Git commit hash of the codebase, and a snapshot of the `git diff HEAD`.
    *   **Environment:** A hash or listing of the `requirements.txt` file, and the CUDA/cuDNN versions.
    *   **Randomness:** The hash of the CHSH settings file, its length, a preview of its first 32 bytes, and all random seeds used (master, noise, bootstrap).
    *   **Hardware:** A basic identifier for the machine the run was performed on.
    *   **Results:** The final fitness score, path to detailed logs, and key diagnostic metrics (S, Δ_NS_A/B, mean correlations, etc.).
    *   **Performance:** Total wall-clock time and estimated FLOPs (e.g., measured in Multiply-Accumulate operations). The formula for FLOPs estimation must be documented. See Appendix B.
*   **Determinism:** The framework should provide a "deterministic mode" that disables non-deterministic GPU kernels to aid in debugging and replication. Post-run integrity checks (re-running a checkpoint to ensure identical output) are recommended for key results.

#### **9.2. Hash-Locked Randomness for Credibility**
*   **Protocol:** For all publication-grade experiments, the binary file containing the random CHSH settings will be generated in advance from a trusted source (e.g., the ANU QRNG). The SHA-256 hash of this file will be pre-published (e.g., in a pre-print abstract or on the project blog) before the final analysis is complete.
*   **Randomness Streams:** All sources of randomness must be explicitly seeded and logged in the run manifest. This includes separate seeds for:
    1.  Substrate A initialization.
    2.  Substrate B initialization.
    3.  Controller weight initialization.
    4.  The optimization algorithm's internal sampling.
    5.  The measurement sampling process in Protocol Q.
*   **Rationale:** This provides cryptographic proof against any accusation of "cherry-picking" a favorable random sequence. It ensures the "test" is fixed and fair for all competing models.

#### **9.3. Multi-Part Publication Plan**
The results of this research program are too diverse for a single publication. A multi-part strategy will be employed to target the appropriate audiences. A consistent figure style (e.g., color scales, axis ordering) will be used across all publications to facilitate comparison.
1.  **Paper I (Control Theory / CompSci):** "Mapping the Computational Cost of Non-Local Classical Correlation." This will present the results of Track 1 (`S(R, K)` surface).
2.  **Paper II (Foundations of Physics):** "A Working Model of the Layered-Time Hypothesis." This will present the results of Track 2 (the successful `S=2.828` emulation) and discuss its philosophical implications.
3.  **Paper III (Information Theory):** "Quantifying Causal Violations." This will present the results of Track 3, analyzing the Absurdist world.
4.  **Software Paper/Tool (Journal of Open Source Software):** "Apsu-Fingerprint: A Tool for Universal Reservoir Characterization." This will present the deliverable from Track 4.
*   **Licensing:** To encourage adoption and reuse, the project will use a permissive licensing scheme: code will be released under the MIT license, while documentation and data will be under CC-BY-4.0.

---

### **10. Glossary of Terms**

*   **Bell-CHSH Inequality:** A mathematical formula used in physics to test whether a system's statistical correlations are classical (`S≤2`) or require a post-classical description (`S>2`).
*   **Classical System:** Any system whose behavior is governed by local realism.
*   **CMA-ES:** A powerful, gradient-free optimization algorithm.
*   **ESN (Echo State Network):** A type of reservoir computer used as our `ClassicalSubstrate`.
*   **Information Ratio (`K`):** A dimensionless parameter `I_controller / I_substrate` measuring the controller's relative complexity.
*   **Layered-Time Hypothesis:** The theory that some physical phenomena might be modeled as emergent properties of a "slow" system being observed and manipulated by a "fast" computational reality.
*   **Non-Signaling Principle:** The physical principle that information cannot be communicated faster than light. Systems with `S≤4` can obey this.
*   **NLC / QIC / UniversalController:** The "fast" controller module that observes and manipulates the substrate.
*   **PR-Box (Popescu-Rohrlich Box):** A theoretical device that achieves the maximum possible non-signaling correlation (`S=4`).
*   **Protocol (Q, M, A):** The specific set of rules and constraints governing the controller for simulating the Quantum, Mannequin, or Absurdist worlds.
*   **Reservoir Computing (RC):** A computational paradigm using the dynamics of a fixed system as a resource.
*   **Speed-ratio `R`:** The dimensionless parameter `τ_slow / τ_fast`, quantifying the controller's speed advantage.
*   **`S(R, K)` Surface:** The primary scientific result of Track 1; a plot of the best attainable Bell score `S` as a function of `R` and `K`.
*   **Tsirelson's Bound:** The theoretical maximum `S` score (`≈ 2.828`) achievable by any system obeying the laws of quantum mechanics.

---

### **11. Known Limitations**

This research program, while rigorous, operates within a simulated environment and has several known limitations that should be acknowledged in any resulting publications:
*   **Simulation Only:** The results are derived from a computational model, not a physical experiment. Claims are about the resources required for *emulation*, not physical instantiation.
*   **No Physical Loopholes:** The simulation is "perfect" in the sense that it is not subject to the physical loopholes (e.g., detection, locality) that must be closed in real-world Bell tests.
*   **Global State Access:** The controller, by design, has access to the full, instantaneous state of the substrate reservoirs. This constitutes a non-local resource that is a core part of the hypothesis being tested.
*   **ReservoirPy Performance:** The current specification relies on `ReservoirPy`, which is primarily a NumPy/CPU library. For experiments requiring high GPU throughput, this may become a bottleneck, and future work might require transitioning to a pure-PyTorch reservoir implementation.

---
### Appendix A: Configuration Schema

This table specifies the required and optional keys for the experiment `config` dictionary.

| Key | Type | Default | Description |
|---|---|---|---|
| `protocol` | str | - | **Required.** 'Mannequin', 'Absurdist', or 'Quantum'. |
| `seed` | int | 42 | Master seed for optimizer and other components. |
| `noise_seed` | int | 42 | Separate seed for the sensor noise RNG. |
| `bootstrap_seed`| int | 42 | Separate seed for the metrics bootstrapping RNG. |
| `controller_delay`| float | 1.0 | The delay `d` used to calculate speed ratio R. |
| `T_total` | int | 4000 | Total simulation steps. Must match length of settings file. |
| `washout_steps`| int | 100 | Initial steps to discard before scoring. Must be multiple of 4. |
| `randomness_file`| str | - | **Required.** Path to the binary CHSH settings file. |
| `binarization_mode`| str | 'sign' | How to convert logits to {-1,+1}. E.g., 'sign', 'bernoulli'. |
| `actuation_scale`| float | 1.0 | Multiplier on controller output before sending to substrate. |
| `substrate_params`| dict | - | **Required.** See sub-table below. |
| `controller_params`| dict | - | **Required.** See sub-table below. |

**`substrate_params`:**
| Key | Type | Default | Description |
|---|---|---|---|
| `N_A`, `N_B` | int | - | **Required.** Number of units in each reservoir. |
| `sr_A`, `sr_B` | float | 0.95 | Spectral radius of each reservoir. |
| `lr_A`, `lr_B` | float | 0.3 | Leaking rate of each reservoir. |
| `noise_A`, `noise_B`| float | 0.0 | Intrinsic noise level in the reservoirs. |
| `seed_A`, `seed_B`| int | 1, 2 | Seeds for initializing reservoir weight matrices. |
| `...` | | | Other reservoirpy hyperparameters... |

**`controller_params`:**
| Key | Type | Default | Description |
|---|---|---|---|
| `N_A`, `N_B` | int | - | **Required.** Must match substrate. |
| `K_controller` | int | 32 | Complexity parameter (e.g., hidden layer size). |
| `R_speed` | float | 1.0 | Speed Ratio (1/d). |
| `signaling_bits` | int | 4 | For Protocol A, the bandwidth of the signaling channel. |
| `internal_cell_config`| dict | `{...}`| Config for the R>1 recurrent cell. See example below. |

**Example `internal_cell_config`:**
```json
{
  "enabled": true,
  "type": "gru",
  "hidden_size": 32
}
```

---
### Appendix B: FLOPs Accounting

To ensure comparable resource metrics across runs, the total Floating Point Operations (FLOPs) will be estimated. We define 1 MAC (Multiply-Accumulate) as 2 FLOPs.

**FLOPs per step:**
`FLOPs_step = FLOPs_substrate + FLOPs_controller`

*   **Substrate (Reservoir):**
    *   `FLOPs_reservoir = 2 * (N_in * N_units + N_units^2)` (Note: Assumes dense weight matrices. If sparse, use `nnz` instead of `N_units^2`).
    *   `FLOPs_substrate = FLOPs_reservoir_A + FLOPs_reservoir_B`

*   **Controller:**
    *   For a Linear layer `(in, out)`: `FLOPs_linear = 2 * in * out`
    *   For a GRUCell `(in, hidden)`: `FLOPs_gru = 3 * (2 * in * hidden + 2 * hidden^2)` (approx.)
    *   `FLOPs_controller_base = sum(FLOPs of all layers in one forward pass)`
    *   If `R > 1` and internal cell enabled: `n_internal = round(R)`
    *   `FLOPs_thinking_loop = (n_internal - 1) * FLOPs_gru`
    *   `FLOPs_controller = FLOPs_controller_base + FLOPs_thinking_loop`

**Total FLOPs for a run:**
`FLOPs_total = FLOPs_step * T_total`