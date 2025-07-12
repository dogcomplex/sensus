# Project Apsu: Phase 1 Review Package (v1)

This package contains the deliverables for the successful completion of **Phase 1: The Null Experiment (Classical Baseline Validation)**.

## Summary

The goal of this phase was to rigorously test our measurement apparatus (`chsh.py` module) and confirm that our baseline `ClassicalSystem` behaves classically, producing an `S` score that does not violate the Bell-CHSH inequality (`S <= 2`).

This was achieved by:
1.  Implementing the `train_readouts` method in `ClassicalSystem` to ensure fair evaluation.
2.  Creating a `chsh.py` module containing the core logic for the CHSH test.
3.  Developing an orchestration script, `run_phase1.py`, to perform the null experiment.
4.  Running the experiment 100 times, where the random CHSH settings were fed directly into the ESNs without a controller.

## Artifacts

*   `phase1_null_experiment_results.png`: The primary deliverable. This plot shows the distribution of the 100 `S` scores obtained.
*   `run_phase1.py`: The exact script used to run the experiment and generate the plot.
*   `chsh.py`: The module defining the CHSH game logic and `S` score calculation.
*   `classical_system.py` (from Phase 0, included for completeness): The underlying classical system that was tested.
*   `requirements.txt`: The list of pinned Python dependencies required to replicate the environment.

## Review Request

Please review the `phase1_null_experiment_results.png` to confirm that it meets the **Phase 1 Success Gate**:
> The mean of the S scores must be ≤ 2.0. No single run should statistically significantly exceed 2.0.

The results show a **mean S-score of ~0.125**, which is well within the classical bound and confirms the validity of our experimental setup. Phase 1 can be considered successfully closed. 