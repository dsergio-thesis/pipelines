
# Astroos LSST/SDSS/HSST Machine Learning PyTorch Pipelines 

## Overview

### AstroOS Pipelines provides astronomy + ML workflows for:

- LSST
- SDSS
- Survey photometry pipelines
- ML-ready dataset generation
- GPU-accelerated training (optional)

### The project uses:

- Python 3.11
- Conda / Mamba environments
- pyproject.toml packaging
- Editable installs
- Optional dev + ML extras
- Optional CUDA support
- CLI interface via `astroos`

------------------------------------------------------------

## Quick Start

Clone:

    git clone <repo-url>
    cd pipelines

Install:

    ./install.sh

Activate environment:

    conda activate astroos-pipelines-0.1.0-py311

Run tests:

    astroos unit_tests


------------------------------------------------------------
## CLI Usage

The installer creates a CLI command:

    astroos <command> [args...]

Commands are auto-detected from:

    $ASTROOS_ROOT/bin/

Examples:

    astroos unit_tests
    astroos download_data
    astroos train_model


------------------------------------------------------------
## Project Structure
```
pipelines/
├── install.sh
├── pyproject.toml
├── Makefile
├── bin/
├── src/
│   ├── astroos_pipelines/
│   └── test/
```

# INSTALLATION GUIDE

## Requirements
------------------------------------------------------------

You must install one of:

- Mambaforge (recommended)
- Miniconda

Verify installation:

    mamba --version
    # or
    conda --version


------------------------------------------------------------
Default Installation (All Platforms)
------------------------------------------------------------

From the project root (pipelines/):

    ./install.sh

This will:

- Create conda env:
      astroos-pipelines-<version>-py311
- Install dependencies
- Install all extras (dev + ml)
- Configure environment variables
- Create CLI command: astroos


------------------------------------------------------------
macOS (CPU-only)
------------------------------------------------------------

Standard install:

    ./install.sh

Activate:

    conda activate astroos-pipelines-0.1.0-py311


Install without extras:

    ./install.sh --no-extras


Install only dev tools:

    ./install.sh --extras=dev


------------------------------------------------------------
Linux (CPU-only)
------------------------------------------------------------

    ./install.sh
    conda activate astroos-pipelines-0.1.0-py311


------------------------------------------------------------
Linux with CUDA (GPU)
------------------------------------------------------------

To install CUDA-enabled PyTorch:

    ./install.sh --cuda

Default CUDA channel:

    cu121

Specify a different CUDA version:

    ./install.sh --cuda=cu118
    ./install.sh --cuda=cu124

What happens:

1) torch + torchvision installed from PyTorch CUDA index
2) Project installed without replacing CUDA wheels


------------------------------------------------------------
RSP (Rubin Science Platform)
------------------------------------------------------------

RSP uses a read-only shared conda stack under /opt.

The installer automatically detects this and redirects
environments to your home directory.

Standard install:

    ./install.sh

If needed manually:

    export MAMBA_ROOT_PREFIX="$HOME/.mamba"
    export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
    ./install.sh


------------------------------------------------------------
Extras Explained
------------------------------------------------------------

Extras defined in pyproject.toml:

dev
    - pytest
    - ruff
    - black

ml
    - torch
    - torchvision

Default install includes:

    [dev,ml]

Override examples:

    ./install.sh --no-extras
    ./install.sh --extras=dev
    ./install.sh --extras=ml


------------------------------------------------------------
Reinstalling
------------------------------------------------------------

Force reinstall dependencies:

    ./install.sh --reinstall


------------------------------------------------------------
Removing the Environment
------------------------------------------------------------

If Makefile exists:

    make clean-env

Or manually:

    mamba env remove -n astroos-pipelines-0.1.0-py311


------------------------------------------------------------
Running Tests
------------------------------------------------------------

After activating environment:

    astroos unit_tests

Or:

    pytest src/test


------------------------------------------------------------
Troubleshooting
------------------------------------------------------------

Read-only filesystem (RSP):

    export MAMBA_ROOT_PREFIX="$HOME/.mamba"
    export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"

CUDA not detected:

    nvidia-smi

Must work before running installer.


------------------------------------------------------------
Python Version
------------------------------------------------------------

This project requires:

    Python 3.11

The installer creates the correct version automatically.


