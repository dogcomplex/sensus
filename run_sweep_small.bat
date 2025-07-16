@echo off
echo "--- Starting Fine-Grained Controller Size Sweep (1-16) ---"

for /L %%s in (1,1,16) do (
    echo "Running experiment with %%s controller units..."
    .venv\\Scripts\\python.exe -m apsu.standalone_experiment --controller-units %%s
)

echo "--- Sweep Complete ---" 