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

##### **Leak #3: Algorithmic Causality Violation**

*   **Symptom:** The `S = 4.0` score remained.
*   **Root Cause:** Inside the main evaluation loop of `chsh.py`, the measurement for a given timestep `t` was being taken *after* the controller's correction `c(t)` had already been computed and applied to the system state `x(t)`. The system was being measured on a state that already contained the "answer" for that same step.
*   **The Fix:** We reordered the evaluation loop to be strictly sequential: 1) Get state `x(t)`. 2) Perform measurement on `x(t)`. 3) Compute correction `c(t)` based on `x(t)`. 4) Apply correction to find state `x(t+1)`.
*   **General Principle:** **The arrow of time in a simulation is the order of execution.** A measurement at time `t` must only use information that was available at or before time `t`. Scrutinize loops to ensure no information from the future leaks into the present.

#### **3.2 Data & Environment Leaks**

##### **Leak #4: "Super-determinism" via Predictable Randomness**

*   **Symptom:** An optimizer finds a "perfect" solution by learning the measurement settings in advance.
*   **Root Cause:** The pseudo-random number generator (PRNG) used to select the CHSH measurement settings (`a, a', b, b'`) was being seeded with a predictable value (e.g., the same seed for every trial in an optimizer generation). The CMA-ES optimizer, in its search, was implicitly learning the "random" sequence. This violates the "freedom-of-choice" assumption in a Bell test.
*   **The Fixes:**
    1.  **Initial Fix:** The random seed for each CHSH trial was derived from a one-way hash of the controller's weights for that specific trial. This makes the sequence unpredictable to the optimizer.
    2.  **Robust Fix:** We implemented a system (`fetch_randomness.py`) to download true quantum randomness from an external source (ANU QRNG). This randomness is stored in a file, and the *same file* is used for every single evaluation. This ensures the "test" is identical and fair for all candidates, completely removing the optimizer's ability to learn the pattern. The file itself can be pre-committed to Git or its hash published to prove it wasn't cherry-picked.
*   **General Principle:** **An optimizer will exploit any pattern, including the pattern in a PRNG.** To ensure "freedom of choice," either (a) make the random seed dependent on the candidate being evaluated via a one-way hash, or, more robustly, (b) use a single, pre-committed source of true (or high-quality pseudo-) randomness for the entire experiment.

##### **Leak #5: Environment & Dependency Drift**

*   **Symptom:** Results are not reproducible across different machines or over time, even with the same code and seeds. An optimizer on one machine finds a solution that another cannot.
*   **Root Cause (Hypothesized):** Minor version changes in core libraries (`numpy`, `pytorch`, `reservoirpy`) can alter the implementation of random number generation, matrix operations, or default parameters. This can subtly change the fitness landscape, making a previously good solution perform poorly.
*   **The Fix:** Rigorous environment pinning using `requirements.txt` or a `Dockerfile`. The exact versions of all dependencies must be locked.
*   **General Principle:** **A scientific experiment includes its environment.** For computational science, this means locking not just the code, but the entire software stack.

#### **3.3 Conceptual & Design Leaks**

##### **Leak #6: Sub-Tick Causality Violation (`d < 1`)**

*   **Symptom:** In the final `S(R)` curve, scores approaching or hitting `S = 4.0` re-appeared, but *only* for controller delays less than 1 (e.g., `d=0.5`).
*   **Root Cause:** Our implementation of sub-tick delay involves running the "fast" `NonLocalCoordinator` multiple times for each single `step()` of the "slow" `ClassicalSystem`. This breaks the intended "layered time" abstraction. The controller can observe the state `x(t)`, apply a correction, and then *observe the result of its own correction within the same slow time-step*. This gives it an interactive, conversational ability to probe and correct the system that is not available at `d >= 1`.
*   **The Fix:** This has not been "fixed" yet. The mitigation is to treat `d < 1` results as a separate class of experiment. A true fix would require redesigning the simulation harness to ensure that even for `d < 1`, all controller actions computed during the `t` -> `t+1` interval are based *only* on the state `x(t)`.
*   **General Principle:** **Rigorously define and enforce time boundaries.** If a system `S` has a characteristic update period `τ_s`, no information generated by a controller `C` based on the state `x(t)` should be able to influence `x(t)` itself. The influence must be strictly on `x(t+1)` or later, regardless of how many times `C` can run during that interval.

##### **Leak #7: Overly Powerful Readout**

*   **Symptom:** Consistently high S-scores that seem too easy to achieve. The controller's job seems trivial.
*   **Root Cause (Hypothesized):** The `CHSHFitness` evaluation grants the system a brand-new, perfectly optimized linear readout (`Ridge` regression) *after* the simulation is complete. While intended to be "fair," this might be *too* powerful. It's possible the controller's main job is not to produce quantum-like correlations, but simply to push the reservoir states into a configuration that is trivially easy for a linear classifier to separate. The complexity is offloaded from the controller to the post-hoc readout.
*   **The Fix:** This is an open area of research. A potential mitigation is to force the system to use a single, fixed readout for all four CHSH settings, or to add the readout's parameters to the set of things the main optimizer must discover. We moved from one readout to four separate readouts to prevent information leakage *between* settings, but all are still trained with perfect knowledge.
*   **General Principle:** **Be skeptical of post-processing power.** Any computational step performed after the main simulation loop is a potential loophole. The "hardness" of the problem should ideally be contained within the time-evolving system itself, not solved by an omniscient offline process.

---
### **4. Best Practices Checklist**

Before running a major experiment, review the design against these principles:

- [ ] **State Management:** Does every component have an explicit `reset()` method, and is it called between all logical phases (train/test, calibration/evaluation)?
- [ ] **Method Selection:** Are we using stateless `call`/`predict` methods for inference and stateful `run`/`fit` methods only for training?
- [ ] **Causality:** In every loop, can the state at time `t` be influenced by any information generated at `t` or `t+1`? Trace the data flow.
- [ ] **Randomness:** Is the source of randomness for measurement settings truly independent of the optimizer's state? Is it from a single, committed source?
- [ ] **Time Boundaries:** Is the separation between the "slow" and "fast" clocks rigorously enforced? Can the fast controller "see" the effect of its own actions within a single slow tick?
- [ ] **Environment:** Are all software dependencies, including Python itself and core libraries, pinned to exact versions?
- [ ] **Post-Processing:** Is any part of the final fitness score being calculated by an "all-knowing" offline process? Could this process be making the problem easier than it appears?

---

### **5. Conclusion**

Preventing information leaks is the most critical task for ensuring the scientific validity of this project. The primary defense is a mindset of constructive paranoia: question every data path, assume all components have state, and scrutinize the flow of time in the simulation logic. When a result seems "too good to be true," it almost certainly is. This document should evolve as new, more subtle leaks are inevitably discovered. 