Please use the /apsu folder for all files related to software development for the APSU project.  The root directory contains additional background documentation, software libraries, and other resources.

---
### Developer Note: Accessing the Environment

To work on this project, you must first activate the Python virtual environment. This ensures you are using the correct dependencies.

From the project root directory, run: `.\.venv\Scripts\activate` (on Windows) or `source .venv/bin/activate` (on macOS/Linux).

All subsequent commands (e.g., `python -m apsu.standalone_blind_experiment`) should be run within this activated shell.
See the main `README.md` for full setup instructions.


------
### Developer Note: Timeouts

Before running any scripts which may take a long time to run, please consider wrapping them in a main.py script top-level caller with a strict timeout set about 3x your expected runtime.  This will ensure that the script will be killed if it runs indefinitely.  It would be good practice to run all scripts in this manner, just in case, with a token 30s max-runtime minimum.

### Developer Note: Logging and Visualizations

Please treat development review and scientific review as separate streams, with development aiming for rapid iteration with high information density, and scientific as "shipped" review packages to a /apsu/review directory organized by phase and version.  These should be designed so that reviewers can asynchronously review the results so far and provide feedback to developers, without slowing down the development process.  Asychronous review for human feedback.  Visualizations, plots, or clear logs would be very appreciated, along with clear context on the code version (perhaps a full copy of the code for that checkpoint?) and a clear description of what was done and why.

### Developer Note: Documentation:

For reservoirpy documentation, please search the `/software/lib/reservoirpy/docs` folder




