# Project Apsu

This repository contains all research, documentation, and software for Project Apsu, an experiment to quantitatively map the relationship between controller/substrate speed ratios and the emergence of non-local statistical correlations.

## Quick Start

### Environment Setup

This project uses a Python virtual environment to manage its dependencies. To get started, follow these steps:

1.  **Create the virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate the environment:**
    *   On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install the required packages:**
    ```bash
    pip install -r apsu/requirements.txt
    ```

### Running Phase 0: Baseline Characterization

With the environment activated and dependencies installed, you can run the first phase of the experiment:

```bash
python -m apsu.classical_system
```

This will execute the diagnostic script and generate a `diagnostics_report.png` file in the `apsu` directory, which visually confirms the health of the baseline classical system. 