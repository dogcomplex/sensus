@echo off
echo "--- Starting DOUBLE-BLIND Controller Size Sweep ---"
echo "This will test for true, generalizable performance."

for %%s in (16 32 48 64 80 96 112 128) do (
    echo "--- Running BLIND experiment with %%s controller units... ---"
    python -m apsu.standalone_blind_experiment --controller-units %%s
)

echo "--- Blind Sweep Complete ---" 