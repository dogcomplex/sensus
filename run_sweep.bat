@echo OFF
echo "--- Starting Controller Size Sweep (Overnight Run - Expanded) ---"

echo "Running experiment with 16 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_16.json

echo "Running experiment with 32 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_32.json

echo "Running experiment with 48 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_48.json

echo "Running experiment with 64 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_64.json

echo "Running experiment with 80 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_80.json

echo "Running experiment with 96 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_96.json

echo "Running experiment with 128 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_128.json

echo "Running experiment with 160 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_160.json

echo "Running experiment with 192 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_192.json

echo "Running experiment with 224 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_224.json

echo "Running experiment with 256 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_256.json

echo "Running experiment with 288 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_288.json

echo "Running experiment with 320 controller units..."
.venv\\Scripts\\python.exe -m apsu.standalone_experiment --config apsu/experiments/standalone_sweep_configs/config_320.json

echo "--- Sweep Complete ---" 