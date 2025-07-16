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
    4.  The Universal Control Parameters
*   **Part 3: Detailed Component Specifications & Implementation**
    5.  The `ClassicalSubstrate` Module
    6.  The `UniversalController` Module
    7.  The `ExperimentHarness` Module
*   **Part 4: The Research Program & Protocols**
    8.  The Multi-Track Research Program
    9.  Data Management, Reproducibility, and Publication Strategy
    10. Glossary of Terms

---

### **PART 1: PROJECT MANDATE & CORE SCIENTIFIC BACKGROUND**

---

### **1. Research Mandate & Evolved Hypothesis**

#### **1.1. High-Level Objective: From "If" to "How Much"**

This document outlines the fourth and most ambitious phase of Project Apsu. The initial phases of this project successfully demonstrated a computational proof-of-concept: a purely classical dynamical system, when guided by a non-local, intelligent controller, can be trained to produce statistical correlations that violate the established bounds of classical physics. The foundational question of *"if this is possible"* within a simulated environment has been answered affirmatively.

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

#### **3.1. Protocol Q: The Quantum World Emulator**

*   **Objective:** To achieve a maximum `S`-score that approaches, but does not exceed, the quantum Tsirelson Bound (`S → 2.828`).
*   **Methodology:** This protocol tests the system's ability to emulate known physics. The `UniversalController` is not a generic, black-box learner. Instead, it is implemented as a **`QuantumInspiredController` (QIC)**. Its internal architecture is explicitly designed to perform the correct mathematics of 2-qubit quantum mechanics.
    *   **Internal State:** The QIC maintains its own internal state vector representing the four complex amplitudes of a 2-qubit system: `[c₀₀, c₀₁, c₁₀, c₁₁]`.
    *   **Hard-Coded Physics:** The core of the controller's logic is a non-trainable, hard-coded implementation of the **tensor product** and **unitary evolution**. When given the measurement settings `(a, b)`, it constructs the correct `4x4` unitary matrix `U_total = U_A(θ_a) ⊗ U_B(θ_b)` and applies it to its internal state: `ψ_next = U_total @ ψ_current`.
    *   **The Learning Task:** The optimizer's job is **not** to discover quantum mechanics. Its much simpler and more tractable task is to learn the "interface" between the ideal quantum simulation and the messy classical substrate. It trains two small neural networks:
        1.  A "sensor" network that learns the best way to update the internal quantum state based on the observed classical reservoir states.
        2.  An "actuator" network that learns the best way to translate the computed quantum outcomes into effective corrective nudges for the classical reservoirs.
*   **Significance:** This protocol provides a powerful baseline. If the resulting `S`-score converges to `~2.828`, it proves our framework is capable of perfectly modeling known quantum physics. It allows us to quantify the "cost of reality"—the resources (`R`, `K`) needed for a classical system to perfectly mimic a true quantum system.

#### **3.2. Protocol M: The Mannequin World (PR-Box) Emulator**

*   **Objective:** To achieve the maximum possible non-signaling correlation (`S → 4.0`).
*   **Methodology:** This protocol tests the absolute limits of a non-local but non-communicating controller. The `UniversalController` is implemented as a generic **`NonLocalCoordinator` (NLC)** (e.g., a standard Multi-Layer Perceptron).
    *   **No Physics Priors:** The NLC has no built-in knowledge of quantum mechanics. It is a "blank slate" universal function approximator.
    *   **Global View:** It receives the states of both reservoirs (`x_A`, `x_B`) and both input settings (`a`, `b`). This global information access is the "non-local" part.
    *   **The Learning Task:** The optimizer's goal is simply to find the weights for the NLC that maximize the final `S`-score. It is free to discover any mathematical trick or correlation rule that achieves this. The expected emergent strategy is the simple PR-Box rule: `x*y = (-1)^(a AND b)`.
    *   **Non-Signaling Constraint:** It is crucial that the computation for Alice's output `y_A` does not have direct access to Bob's input `b`, and vice-versa. The controller has global knowledge, but its output channels must be causally separated to respect the non-signaling principle.
*   **Significance:** This protocol allows us to explore "super-quantum" correlations. It tests the computational power of a system that is freed from the specific mathematical constraints of quantum mechanics but not from causality. The resources (`R`, `K`) required to reach `S=4` can be directly compared to those required to reach `S=2.828` in Protocol Q.

#### **3.3. Protocol A: The Absurdist World (Signaling) Emulator**

*   **Objective:** To demonstrate that `S > 4.0` is achievable and to quantify the relationship between communication bandwidth and the degree of this "super-causal" correlation.
*   **Methodology:** This protocol explicitly and deliberately violates the Non-Signaling principle. It serves as a control experiment to validate our understanding of the system's absolute limits.
    *   **Explicit Signaling Channel:** The `UniversalController` is architected to create a direct communication channel. The logic for computing Bob's corrective action `c_B` will receive Alice's input setting `a` as an explicit input argument.
    *   **The Learning Task:** The optimizer will now learn to exploit this direct channel. It will likely learn a very simple policy: Alice computes her output, sends it to Bob via the signaling channel, and Bob outputs the same value, guaranteeing perfect correlation (`E=1`) in all cases except one, leading to `S > 4`.
    *   **The Independent Variable:** In this protocol, we will introduce a new "knob": the **bandwidth of the signaling channel**. We can constrain the information passed from Alice's side to Bob's to a single bit, 2 bits, 4 bits, etc., by quantizing the signal.
*   **Significance:** This protocol is not intended to model a plausible physical reality. Its purpose is to act as a **diagnostic tool**. It allows us to create a plot of `S_max` vs. "Bits of Signaling," which provides a concrete, information-theoretic grounding for the entire framework. It demonstrates that we understand the system's behavior so well that we can make it break any physical bound by a controllable amount.

---

### **4. The Universal Control Parameters**

To make our results generalizable and comparable across different physical systems, we define our experimental "knobs" as dimensionless parameters.

#### **4.1. Speed Ratio (`R`)**
*   **Definition:** `R = τ_slow / τ_fast`, where `τ_slow` is the characteristic update time of the substrate and `τ_fast` is the end-to-end latency of the controller.
*   **Significance:** This parameter quantifies the fundamental "superpower" of the layered-time architecture. It measures how many computational "thoughts" the controller can have for every single "moment" the substrate experiences.
*   **Implementation:** In our simulation, we fix `τ_slow = 1` (a single `step()` of the ESN). We will implement `τ_fast` by controlling an integer **delay parameter `d`**.
    *   `d ≥ 1`: This corresponds to a lagged controller (`R ≤ 1`). The corrective signal `c(k)` computed from state `x(k)` is applied at a future step `k+d`.
    *   `d < 1` (e.g., `d=0.5`): This corresponds to a super-fast controller (`R > 1`). This will be implemented via the corrected **internal loop protocol** (see Section 5.3 of the `v3.0` document), where the controller performs `1/d` computations based only on state `x(k)` before a single correction is applied at `k+1`. This robustly tests the "more computation time, not more information" hypothesis.
*   **Target Test Set:** The primary `S(R)` curve will be generated by sweeping `d` across a logarithmic-like scale, e.g., `{0.25, 0.5, 1, 2, 4, 8, 16}`.

#### **4.2. Information Ratio (`K`)**
*   **Definition:** `K = I_controller / I_substrate`, where `I` represents the information capacity of a component.
*   **Significance:** This parameter quantifies the "intelligence" or "complexity" of the controller relative to the system it is controlling. The results of our `goldilocks_sweep` suggest that the relationship between `K` and performance is highly non-monotonic, with a distinct "Goldilocks Zone" of optimal complexity.
*   **Implementation:**
    *   `I_substrate`: The number of state variables, `2 * N` (for two ESNs).
    *   `I_controller`: The number of trainable weights and biases in the `UniversalController`'s neural network.
*   **Target Test Set:** We will generate multiple `S(R)` curves, each for a different value of `K`. This will be done by running the experiment with different controller architectures (e.g., hidden layers of size {16, 32, 64, 128}). This will produce a 3D surface plot, `S(R, K)`, as the main deliverable of Track 1.

#### **4.3. Sensor Noise (`σ_noise`)**
*   **Definition:** The standard deviation of zero-mean Gaussian noise that is added to the substrate's state vector *before* it is passed to the controller.
    `input_to_controller = x_substrate + N(0, σ_noise)`
*   **Significance:** This parameter models the real-world imperfection of measurement devices. It allows us to test the robustness of our learned control policies. A truly powerful controller should be able to function even with noisy, incomplete information.
*   **Implementation:** This will be a configurable parameter in the `ExperimentHarness`. We will run dedicated experiments (e.g., as part of the robustness checks in the final phase) to plot `S_max` as a function of `σ_noise` for a fixed, high-performing `(R, K)` configuration.

---

### **PART 3: DETAILED COMPONENT SPECIFICATIONS & IMPLEMENTATION**


This section provides the detailed, actionable specifications for the three core software modules of the Project Apsu framework: the `ClassicalSubstrate`, the `UniversalController`, and the `ExperimentHarness`. The design emphasizes modularity, configurability, and robust, defensive programming to ensure the scientific integrity and reproducibility of the results.

### **5. The `ClassicalSubstrate` Module**

*   **Filename:** `apsu/substrate.py`
*   **Purpose:** This module represents the physical "slow medium" being controlled. It encapsulates the classical dynamical systems, their state, and their response to external inputs. Its design must ensure that the two subsystems (A and B) are computationally independent and that their state can be reliably reset and diagnosed.

#### **5.1. Class Definition: `ClassicalSubstrate`**

```python
import reservoirpy as rpy
import numpy as np
import torch

class ClassicalSubstrate:
    """
    Encapsulates the two Echo State Network (ESN) reservoirs representing the
    "slow medium" of the experiment (Systems A and B).
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
            device (torch.device): The compute device ('cpu' or 'cuda').
        """
        self.device = device
        self.N_A, self.N_B = N_A, N_B

        # Use EchoTorch for GPU acceleration. Each reservoir is a separate instance.
        # Rationale: Using distinct seeds ensures there is no accidental correlation
        # in their internal random wiring.
        self.reservoir_A = rpy.nodes.Reservoir(units=N_A, sr=sr_A, lr=lr_A, noise_rc=noise_A, seed=seed_A).to(device)
        self.reservoir_B = rpy.nodes.Reservoir(units=N_B, sr=sr_B, lr=lr_B, noise_rc=noise_B, seed=seed_B).to(device)

        # The state vectors are stored as PyTorch tensors for GPU operations.
        self.state_A = torch.zeros(1, N_A, device=device)
        self.state_B = torch.zeros(1, N_B, device=device)

    def step(self, input_A, input_B):
        """
        Evolves the system by one time step.

        Args:
            input_A (torch.Tensor): The input signal for reservoir A.
            input_B (torch.Tensor): The input signal for reservoir B.

        Returns:
            (torch.Tensor, torch.Tensor): The new state vectors x_A(t+1), x_B(t+1).
        """
        # .run() is stateful in reservoirpy, we call the underlying forward pass
        # logic for stateless evolution managed by the harness.
        self.state_A = self.reservoir_A.call(input_A, self.state_A)
        self.state_B = self.reservoir_B.call(input_B, self.state_B)
        return self.state_A, self.state_B

    def reset(self):
        """
        Resets the internal state of both reservoirs to zero.
        This is a CRITICAL method for ensuring trial independence.
        """
        self.state_A = torch.zeros(1, self.N_A, device=self.device)
        self.state_B = torch.zeros(1, self.N_B, device=self.device)

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

*   **Filename:** `apsu/controller.py`
*   **Purpose:** This is the heart of the "fast medium." It is a polymorphic class designed to be configurable to run any of the three experimental protocols (Quantum, Mannequin, Absurdist). It takes the state of the substrate as input and produces corrective actions.

#### **6.1. Class Definition: `UniversalController`**

```python
import torch
import torch.nn as nn

class UniversalController(nn.Module):
    """
    A polymorphic controller that can be configured to emulate different
    physical realities (Quantum, Mannequin, Absurdist).
    """
    def __init__(self, protocol, N_substrate, K_controller, R_speed, device):
        """
        Initializes the controller for a specific protocol.

        Args:
            protocol (str): 'Quantum', 'Mannequin', or 'Absurdist'.
            N_substrate (int): The dimension of the substrate state space (e.g., N_A + N_B).
            K_controller (int): A parameter controlling the controller's complexity (e.g., hidden layer size).
            R_speed (float): The speed ratio, used to configure internal loop iterations for d < 1.
        """
        super().__init__()
        self.protocol = protocol
        self.device = device
        self.R = R_speed

        # Build the neural network architecture based on the protocol
        self._build_network(N_substrate, K_controller)

    def _build_network(self, N_substrate, K_controller):
        """Helper method to construct the appropriate NN architecture."""
        if self.protocol == 'Quantum':
            # Protocol Q: A structured network with hard-coded physics
            self.quantum_state_dim = 4 # For a 2-qubit system
            # NN to map classical states to quantum perturbations
            self.sensor_head = nn.Sequential(...)
            # NN to map quantum results to classical corrections
            self.actuator_head = nn.Sequential(...)
            # The internal quantum state will be a tensor managed in the forward pass
            self.internal_quantum_state = torch.tensor([1/sqrt(2), 0, 0, 1/sqrt(2)], device=self.device)

        elif self.protocol == 'Mannequin' or self.protocol == 'Absurdist':
            # Protocol M & A: A generic MLP universal function approximator
            # The architecture size is determined by K_controller
            self.mlp = nn.Sequential(
                nn.Linear(N_substrate, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, 2) # Output for c_A and c_B
                nn.Tanh() # Bounded output for stability
            ).to(self.device)

            if self.protocol == 'Absurdist':
                # For Protocol A, we need to modify the forward pass to accept
                # the signaling information. The architecture can be the same.
                pass
    
    def forward(self, x_A, x_B, settings_A=None, settings_B=None):
        """The main forward pass, dispatched based on protocol."""
        
        # Internal loop for R > 1 (d < 1) to simulate "more thinking time"
        internal_iterations = max(1, int(self.R))
        
        # --- The core logic fork based on the simulated world ---
        
        if self.protocol == 'Quantum':
            # Pseudocode for Protocol Q
            # ... (as detailed in Section 3.1) ...
            # 1. Use sensor_head to compute perturbation from x_A, x_B
            # 2. Apply perturbation to self.internal_quantum_state
            # 3. For 'internal_iterations' times:
            #    a. Apply hard-coded quantum gate U(settings_A, settings_B)
            #    b. Update internal_quantum_state
            # 4. Use actuator_head to compute correction_signal from final quantum state
            # 5. Return correction_signal
            
        elif self.protocol == 'Mannequin':
            # Pseudocode for Protocol M
            # The input to the MLP is the full non-local state
            controller_input = torch.cat([x_A, x_B, settings_A, settings_B], dim=-1) # Global info
            # For 'internal_iterations' times:
            #    a. Pass controller_input through an internal recurrent cell (optional)
            # 4. Final MLP pass computes the correction
            correction_signal = self.mlp(controller_input)
            return correction_signal
            
        elif self.protocol == 'Absurdist':
            # Pseudocode for Protocol A
            # Compute Alice's correction based on her local view
            correction_A = self.mlp_A(torch.cat([x_A, settings_A], dim=-1))
            
            # THIS IS THE CHEAT: Bob's computation gets Alice's setting
            signaling_info = settings_A 
            correction_B = self.mlp_B(torch.cat([x_B, settings_B, signaling_info], dim=-1))
            return torch.cat([correction_A, correction_B], dim=-1)

    def reset(self):
        """Resets any internal state of the controller."""
        if self.protocol == 'Quantum':
            # Reset to the initial Bell state
            self.internal_quantum_state = torch.tensor(...)
```
*   **Defensive Design:** The class is explicitly polymorphic. The choice of which "universe" to simulate is a high-level configuration parameter, preventing accidental mixing of protocols. The use of separate network heads (`sensor_head`, `actuator_head`) for Protocol Q makes the learned components modular and analyzable. The `Tanh` output on the MLP provides crucial stability.

---

### **7. The `ExperimentHarness` Module**

*   **Filename:** `apsu/harness.py`
*   **Purpose:** This is the main orchestration script. It is the "lab bench" that sets up the experiment, runs the simulation loop, gathers data, and computes the final fitness score.

#### **7.1. Class Definition: `ExperimentHarness`**

```python
class ExperimentHarness:
    def __init__(self, config):
        """
        Initializes the entire experiment from a configuration dictionary.
        
        Args:
            config (dict): A dictionary specifying all parameters for the
                           substrate, controller, protocol, and optimization.
        """
        self.config = config
        self.substrate = ClassicalSubstrate(config.substrate_params)
        self.controller = UniversalController(config.controller_params)
        # Load pre-generated, cryptographically secure randomness
        self.chsh_settings = load_randomness(config.randomness_file)

    def evaluate_fitness(self, controller_weights):
        """
        Performs one full fitness evaluation for a given set of controller weights.
        This is the function that the GlobalOptimizer will call repeatedly.
        """
        # 1. SETUP
        self.substrate.reset()
        self.controller.reset()
        self.controller.load_weights(controller_weights)
        
        # Introduce sensor noise if specified in config
        sensor_noise_std = self.config.get('sensor_noise', 0.0)

        # 2. SIMULATION LOOP
        outputs_A, outputs_B = [], []
        for t in range(self.config.T_total):
            state_A, state_B = self.substrate.get_state()
            
            # Apply sensor noise
            noisy_state_A = state_A + torch.randn_like(state_A) * sensor_noise_std
            noisy_state_B = state_B + torch.randn_like(state_B) * sensor_noise_std
            
            # Get settings for this time step
            setting_A, setting_B = self.chsh_settings[t]
            
            # Compute and apply correction (respecting delay d)
            # The harness manages the delay buffer
            correction = self.controller.forward(noisy_state_A, noisy_state_B, setting_A, setting_B)
            delayed_correction = self.delay_buffer.push(correction)
            
            self.substrate.step(setting_A + delayed_correction_A, setting_B + delayed_correction_B)

            # Record outputs
            # ... (using the appropriate honest readout method)

        # 3. SCORING
        S_score = calculate_chsh_score(outputs_A, outputs_B, self.chsh_settings)
        
        # 4. DIAGNOSTICS
        diagnostics = self.compute_diagnostics(...) # Rich data packet
        
        return diagnostics

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
*   **Scientific Rationale:** The choice of optimizer is not neutral; different algorithms have different inductive biases and excel on different types of problems. A "brute-force" approach with a generic optimizer may be computationally wasteful. This track aims to find the "right tool for the job."
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
    1.  **Protocol:** This track will exclusively use **Protocol M (The Mannequin World Emulator)**. The `UniversalController` will be a generic MLP, free to discover the optimal correlation strategy.
    2.  **Grid Search:** A grid search will be performed over the two primary independent variables:
        *   **Speed Ratio (`R`):** Sweeping the delay parameter `d` across `{0.25, 0.5, 1, 2, 4, 8, 16}`.
        *   **Information Ratio (`K`):** Sweeping the controller complexity by using NLCs with hidden layer sizes of `{8, 16, 32, 64, 128}`.
    3.  **Optimization:** For each `(R, K)` point on the grid, the `GlobalOptimizer` (selected from Track 0) will be run for a full, long-duration optimization (e.g., 1000 generations) to find the maximum `S`-score.
*   **Deliverable:**
    1.  A 3D surface plot of `S_max(R, K)`, visualizing the trade-offs between controller speed and complexity.
    2.  A 2D contour plot identifying the "Goldilocks Zone"—the region of `(R, K)` space that yields the highest performance. This plot will be the central figure of our first major publication.

#### **8.3. Track 2: The Limits of Quantum Emulation (Executing Protocol Q)**

*   **Goal:** To determine if our framework can successfully reproduce the precise statistical limits of known quantum physics (i.e., the Tsirelson Bound).
*   **Methodology:**
    1.  **Protocol:** This track will exclusively use **Protocol Q (The Quantum World Emulator)**. The `UniversalController` will be the physics-informed `QuantumInspiredController` with hard-coded quantum mathematics.
    2.  **Optimization:** The optimizer will find the optimal parameters for the "sensor" and "actuator" sub-networks of the QIC, using the best `(R, K)` configuration discovered in Track 1.
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
*   **Experiment IDs:** Every experiment run will be assigned a unique identifier (UUID). All artifacts (logs, plots, saved models, configurations) will be stored in a directory named with this UUID.
*   **Version Control:** The entire codebase will be managed with Git. The Git commit hash will be logged for every experiment run.
*   **Environment Pinning:** A locked `requirements.txt` (or similar) and a `Dockerfile` will be maintained to ensure the exact software environment can be replicated perfectly. For GPU-dependent operations, the exact CUDA driver version will be logged.

#### **9.2. Hash-Locked Randomness for Credibility**
*   **Protocol:** For all publication-grade experiments, the binary file containing the random CHSH settings will be generated in advance from a trusted source (e.g., the ANU QRNG). The SHA-256 hash of this file will be pre-published (e.g., in a pre-print abstract or on the project blog) before the final analysis is complete.
*   **Rationale:** This provides cryptographic proof against any accusation of "cherry-picking" a favorable random sequence. It ensures the "test" is fixed and fair for all competing models.

#### **9.3. Multi-Part Publication Plan**
The results of this research program are too diverse for a single publication. A multi-part strategy will be employed to target the appropriate audiences.
1.  **Paper I (Control Theory / CompSci):** "Mapping the Computational Cost of Non-Local Classical Correlation." This will present the results of Track 1 (`S(R, K)` surface).
2.  **Paper II (Foundations of Physics):** "A Working Model of the Layered-Time Hypothesis." This will present the results of Track 2 (the successful `S=2.828` emulation) and discuss its philosophical implications.
3.  **Paper III (Information Theory):** "Quantifying Causal Violations." This will present the results of Track 3, analyzing the Absurdist world.
4.  **Software Paper/Tool (Journal of Open Source Software):** "Apsu-Fingerprint: A Tool for Universal Reservoir Characterization." This will present the deliverable from Track 4.

---

### **10. Glossary of Terms**

*   **Bell-CHSH Inequality:** A mathematical formula used in physics to test whether a system's statistical correlations are classical (`S≤2`) or require a quantum description (`S>2`).
*   **Classical System:** Any system whose behavior is governed by local realism.
*   **CMA-ES:** A powerful, gradient-free optimization algorithm.
*   **ESN (Echo State Network):** A type of reservoir computer used as our `ClassicalSubstrate`.
*   **Information Ratio (`K`):** A dimensionless parameter `I_controller / I_substrate` measuring the controller's relative complexity.
*   **Layered-Time Hypothesis:** The theory that quantum mechanics is an emergent phenomenon experienced by "slow" observers being manipulated by a "fast" computational reality.
*   **Non-Signaling Principle:** The physical principle that information cannot be communicated faster than light. Systems with `S≤4` can obey this.
*   **NLC / QIC / UniversalController:** The "fast" controller module that observes and manipulates the substrate.
*   **PR-Box (Popescu-Rohrlich Box):** A theoretical device that achieves the maximum possible non-signaling correlation (`S=4`).
*   **Protocol (Q, M, A):** The specific set of rules and constraints governing the controller for simulating the Quantum, Mannequin, or Absurdist worlds.
*   **Reservoir Computing (RC):** A computational paradigm using the dynamics of a fixed system as a resource.
*   **Speed-ratio `R`:** The dimensionless parameter `τ_slow / τ_fast`, quantifying the controller's speed advantage.
*   **`S(R, K)` Surface:** The primary scientific result of Track 1; a plot of the best attainable Bell score `S` as a function of `R` and `K`.
*   **Tsirelson's Bound:** The theoretical maximum `S` score (`≈ 2.828`) achievable by any system obeying the laws of quantum mechanics.