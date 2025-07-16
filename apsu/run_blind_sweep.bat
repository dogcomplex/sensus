@echo off
setlocal

set VENV_PATH=.\.venv\Scripts\python.exe

ECHO Running sweep of controller sizes 1-16 with DOUBLE BLIND protocol.
ECHO This will take a significant amount of time.

for %%i in (8 16 32 48 64 80 96 112 128) do (
    echo.
    echo -------------------------------------------------
    echo Running BLIND sweep for controller size %%i
    echo -------------------------------------------------
    %VENV_PATH% -m apsu.standalone_blind_experiment --controller-units %%i
)

echo.
echo -------------------------------------------------
echo Blind sweep complete.
echo -------------------------------------------------
endlocal 