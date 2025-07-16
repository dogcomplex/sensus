@echo OFF
setlocal

set CONTROLLER_UNITS=64
set DELAYS=1 2 3 5 8 13

echo "Running blind sweep for controller size %CONTROLLER_UNITS% over delays: %DELAYS%"

for %%d in (%DELAYS%) do (
    echo "-----------------------------------------------------"
    echo "Running experiment with delay: %%d"
    echo "-----------------------------------------------------"
    python -m apsu.standalone_blind_experiment --controller-units %CONTROLLER_UNITS% --delay %%d
    if errorlevel 1 (
        echo "Experiment failed for delay %%d. Stopping sweep."
        exit /b 1
    )
)

echo "Sweep complete." 