# Project Apsu: Information Leakage Prevention Guide

**Document Version:** 1.1
**Date:** July 14, 2025
**Status:** Living Document

---

### **1. Introduction**

This document serves as a comprehensive guide to the various forms of information leakage discovered and patched during the development of Project Apsu. The project's core hypothesis relies on a rigorous separation between a "slow" classical system and a "fast" non-local controller. An information leak, no matter how subtle, can invalidate experimental results by providing the controller with an unfair advantage, allowing it to "cheat" the CHSH test and produce scientifically invalid scores (e.g., `S > 2.828`).

This guide is intended for all developers on this project. It should be consulted before any major redesign of the `ClassicalSystem`, `NonLocalCoordinator`, or the `CHSHFitness` evaluation harness to prevent the re-introduction of these hard-won lessons. A "leak" is any pathway, whether in code, data, or experimental design, that violates the core assumptions of the CHSH game—specifically, locality and freedom of choice.

---

### **2. Leak Taxonomy**

We categorize leaks into three main types:
*   **Implementation-Level Leaks:** Bugs in the code that violate causality or improperly handle state.
*   **Data & Environment Leaks:** Issues related to the data used for the experiment, particularly randomness, and the environment it runs in.
*   **Conceptual & Design Leaks:** Flaws in the experimental design itself that create loopholes, even if the code is bug-free.

---

### **3. Catalogue of Discovered Leaks**

#### **3.1 Implementation-Level Leaks**

##### **Leak #1: Implicit Stateful Carryover (Reservoir)**

*   **Symptom:** The CHSH score instantly and consistently hit the impossible value of `S = 4.0`.
*   **Root Cause:** The `ClassicalSystemReservoirPy` class, which wraps the `reservoirpy` nodes, did not have an explicit `reset()` method. The internal state of the ESN reservoirs (`self.hidden`) from the calibration/training phase was being carried over and used as the starting state for the subsequent evaluation/testing phase. The system was effectively being tested on the same data it was trained on.
*   **The Fix:** We implemented a `reset()` method in `ClassicalSystemReservoirPy` that explicitly calls the `.reset()` method on the underlying `reservoirpy` `Reservoir` nodes, ensuring a clean slate between experimental phases.
*   **General Principle:** **Assume all components are stateful unless proven otherwise.** Always implement and call explicit `reset()` functions between logically distinct phases of an experiment (e.g., `train`, `test`, `calibrate`, `evaluate`). Never assume an object's state is automatically cleared.

##### **Leak #2: Implicit Stateful Carryover (Readout)**

*   **Symptom:** Even after fixing the reservoir state leak, the `S = 4.0` score persisted.
*   **Root Cause:** The `Ridge` readout nodes from `reservoirpy` also have stateful behavior. When we called the `.run()` method during the "live" evaluation phase, it was subtly updating the readout's internal state (e.g., covariance matrices used for fitting). This meant information from earlier evaluation blocks was leaking into the readout's behavior on later blocks.
*   **The Fix:** We switched from using the stateful `.run()` method to the stateless `.call()` method for the readout nodes during the live evaluation phase. The `.call()` method simply applies the already-trained linear transformation without updating the node's internal state.
*   **General Principle:** **Differentiate between training and inference methods.** Library methods like `.fit()`, `.train()`, or `.run()` often imply state updates. For pure evaluation on unseen data, always use a stateless prediction method like `.predict()`, `.call()`, or `.transform()`.

##### **Leak #3: Algorithmic Causality Violation (Sequential)**
*   **Symptom:** The `S = 4.0` score remained, even with state being reset.
*   **Root Cause:** Inside a *sequential, step-by-step* evaluation loop, the measurement for a given timestep `t` was being taken *after* the controller's correction `c(t)` had already been computed and applied to the system state `x(t)`. The system was being measured on a state that already contained the "answer" for that same step.
*   **The Fix:** We reordered the evaluation loop to be strictly sequential: 1) Get state `x(t)`. 2) Perform measurement on `x(t)`. 3) Compute correction `c(t)` based on `x(t)`. 4) Apply correction to find state `x(t+1)`.
*   **General Principle:** **The arrow of time in a simulation is the order of execution.** A measurement at time `t` must only use information that was available at or before time `t`. Scrutinize loops to ensure no information from the future leaks into the present.

##### **Leak #4: Vectorized Causality Violation (Post-Hoc Evaluation)**
*   **Symptom:** `S`-scores that are consistently and inexplicably high (e.g., `S > 2.8`), often discovered after a major performance-enhancing refactor to a "single-pass" or "fully vectorized" design.
*   **Root Cause:** This is a severe architectural leak where the simulation harness abandons a sequential, step-by-step loop for performance reasons. Instead, it first computes a baseline history of the system's evolution *without any controller input*. It then feeds this *entire time-series history* to the controller in a single forward pass. The controller sees the entire "movie" of the universe from start to finish and computes all its corrective actions at once.
*   **The Fix:** A complete refactoring back to a sequential, step-by-step simulation is required. The `evaluate_fitness` method must contain a `for t in range(T_total):` loop that strictly enforces the causal order of operations: get state `x(t)`, compute correction `c(t)`, apply correction to get `x(t+1)`.
*   **General Principle:** **Performance optimizations must not break causality.** A controller must be a controller (acting sequentially in time), not a post-hoc analysis engine. If the controller's `forward()` method accepts an entire time series as input, it is almost certainly a leaky design.

#### **3.2 Data & Environment Leaks**

##### **Leak #5: "Super-determinism" via Predictable Randomness**
*   **Symptom:** An optimizer finds a "perfect" solution by learning the measurement settings in advance, or performance varies wildly and inexplicably between runs.
*   **Root Cause:** The pseudo-random number generator (PRNG) used to select the CHSH measurement settings (`a, b`) is predictable. This can happen in two ways:
    1.  **Fixed Seed:** The PRNG is seeded with the same value for every trial. The optimizer quickly learns the repeating sequence.
    2.  **Candidate-Dependent Seed:** The PRNG seed is derived from the candidate being evaluated (e.g., `hash(weights)`). While this prevents prediction, it creates a different "test" for every candidate, making the fitness landscape noisy and non-stationary.
*   **The Fix (Gold Standard):** Use a single, pre-committed source of true (or high-quality, cryptographically secure) randomness for the entire experiment. This is achieved by:
    1.  Generating a binary file of random CHSH settings in advance (`generate_chsh_settings.py`).
    2.  Committing this file to the repository. Publishing the file's hash provides cryptographic proof that the test was not altered.
    3.  Having every single fitness evaluation read from this *exact same file*. This ensures the "test" is identical and fair for all candidates, making the search for a high score a well-posed problem.
*   **General Principle:** **Treat the measurement settings as a fixed, universal test, not a random variable.** An optimizer will exploit any pattern or statistical advantage in the test itself. A single, committed randomness file is the strongest defense.

##### **Leak #6: Environment & Dependency Drift**
*   **Symptom:** Results are not reproducible across different machines or over time, even with the same code and seeds. An optimizer on one machine finds a solution that another cannot.
*   **Root Cause (Hypothesized):** Minor version changes in core libraries (`numpy`, `pytorch`, `reservoirpy`) can alter the implementation of random number generation, matrix operations, or default parameters. This can subtly change the fitness landscape, making a previously good solution perform poorly.
*   **The Fix:** Rigorous environment pinning using `requirements.txt` or a `Dockerfile`. The exact versions of all dependencies must be locked and installed for any replication attempt.
*   **General Principle:** **A scientific experiment includes its environment.** For computational science, this means locking not just the code, but the entire software stack.

#### **3.3 Conceptual & Design Leaks**

##### **Leak #7: Sub-Tick Causality Violation (`d < 1`)**
*   **Symptom:** In the final `S(R)` curve, scores approaching or hitting `S = 4.0` re-appeared, but *only* for controller delays less than 1 (e.g., `d=0.5`).
*   **Root Cause (Incorrect Implementation):** An implementation of sub-tick delay that involves running the "fast" `NonLocalCoordinator` multiple times for each single `step()` of the "slow" `ClassicalSystem` in an *interactive* way. This breaks the intended "layered time" abstraction. The controller can observe the state `x(t)`, apply a correction, and then *observe the result of its own correction within the same slow time-step*. This gives it an interactive, conversational ability to probe and correct the system that is not available at `d >= 1`.
*   **The Fix (Correct Implementation):** The "thinking loop" must be implemented as a recurrent operation on a *static* view of the state `x(t)`.
    1.  The harness gets the state `x(t)` once.
    2.  The harness runs a `for _ in range(R):` loop.
    3.  Inside the loop, the controller's `forward()` method is called repeatedly. It takes `x(t)` as input, but also its *own* internal hidden state from the previous iteration. It outputs a new hidden state.
    4.  After the loop, the final output of the controller is used to compute the correction for the *next* substrate step, `x(t+1)`.
*   **General Principle:** **Rigorously define and enforce time boundaries.** If a system `S` has a characteristic update period `τ_s`, no information generated by a controller `C` based on the state `x(t)` should be able to influence `x(t)` itself. The influence must be strictly on `x(t+1)` or later, regardless of how many times `C` can run during that interval.

##### **Leak #8: Overly Powerful Readout**
*   **Symptom:** Consistently high S-scores that seem too easy to achieve. The controller's job seems trivial.
*   **Root Cause (Hypothesized):** The `CHSHFitness` evaluation grants the system a brand-new, perfectly optimized linear readout (`Ridge` regression) *after* the simulation is complete. While intended to be "fair," this might be *too* powerful. It's possible the controller's main job is not to produce quantum-like correlations, but simply to push the reservoir states into a configuration that is trivially easy for a linear classifier to separate. The complexity is offloaded from the controller to the post-hoc readout.
*   **The Fix:** This is an open area of research. A potential mitigation is to force the system to use a single, fixed readout for all four CHSH settings, or to add the readout's parameters to the set of things the main optimizer must discover. We moved from one readout to four separate readouts to prevent information leakage *between* settings, but all are still trained with perfect knowledge. The most robust design (which we now use) is an end-to-end model where the controller's final layer *is* the readout, and its weights are part of the optimization.
*   **General Principle:** **Be skeptical of post-processing power.** Any computational step performed after the main simulation loop is a potential loophole. The "hardness" of the problem should ideally be contained within the time-evolving system itself, not solved by an omniscient offline process.

---

### **4. Advanced & Theoretical Leaks**

This section catalogues more subtle, second-order leak pathways. While less likely to occur in practice with the current architecture, they are important to consider for future hardware implementations or more complex software designs.

-   **Optimizer Side-Channels:** In principle, the optimization algorithm itself could be a source of leaks. For example, if a candidate-dependent random seed is used (`hash(weights)`), the optimizer could theoretically learn to produce candidate weights that cause hash collisions, allowing it to re-run evaluations on a "lucky" CHSH sequence it has seen before. Using a single, committed randomness file for all evaluations is the most effective defense against this.
-   **Hardware & Concurrency Side-Channels:** When running multiple evaluations in parallel (e.g., with `torch.multiprocessing`), there is a theoretical possibility of crosstalk through shared hardware resources like CPU/GPU caches or memory controllers. Furthermore, some CUDA operations are non-deterministic by default, which can affect reproducibility. Using the `spawn` start method for multiprocessing and explicitly setting deterministic algorithm flags (`torch.use_deterministic_algorithms(True)`) are the primary mitigations.
-   **Floating-Point Precision Side-Channels:** In a mixed-precision system, information could theoretically be encoded in the least significant bits of a number, which are then truncated or rounded in a state-dependent way by a lower-precision component. This is not a significant concern in the current, standardized `float32` pipeline but is a known consideration in FPGA and neuromorphic hardware design.

---

### **5. Best Practices Checklist**

Before running a major experiment, review the design against these principles:

- [ ] **State Management:** Does every component have an explicit `reset()` method, and is it called between all logical phases (train/test, calibration/evaluation)?
- [ ] **Causality & Vectorization:** Is the simulation a sequential `for` loop over time? Does the controller ever see the entire time-series history at once?
- [ ] **Method Selection:** Are we using stateless `call`/`predict` methods for inference and stateful `run`/`fit` methods only for training?
- [ ] **Randomness:** Is the source of randomness for measurement settings from a single, pre-committed, high-quality randomness file?
- [ ] **Time Boundaries (`d<1`):** Is the "thinking loop" for a fast controller implemented as a recurrent update on a static snapshot of the past state?
- [ ] **Environment:** Are all software dependencies, including Python itself and core libraries, pinned to exact versions in a `requirements.txt` file?
- [ ] **Post-Processing:** Is any part of the final fitness score being calculated by an "all-knowing" offline process?

---

### **6. Conclusion**

Preventing information leaks is the most critical task for ensuring the scientific validity of this project. The primary defense is a mindset of constructive paranoia: question every data path, assume all components have state, and scrutinize the flow of time in the simulation logic. When a result seems "too good to be true," it almost certainly is. This document should evolve as new, more subtle leaks are inevitably discovered. 