@echo off
echo "--- Starting Cross-Seed Robustness Test ---"

set SEEDS=42 101 1337 2024 7777
set SIZES=1 3

for %%s in (%SIZES%) do (
    for %%r in (%SEEDS%) do (
        echo "Running experiment with %%s controller units and seed %%r..."
        .venv\\Scripts\\python.exe -m apsu.standalone_experiment --controller-units %%s --seed %%r
    )
)

echo "--- Seed Sweep Complete ---" 