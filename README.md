
# Astroos LSST/SDSS/HSST Machine Learning PyTorch Pipelines 

## Overview

### AstroOS Pipelines provides astronomy + ML workflows for:

- LSST
- SDSS
- Survey photometry pipelines
- ML-ready dataset generation

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


------------------------------------------------------------
Default Installation (All Platforms)
------------------------------------------------------------

From the project root (pipelines/):

    ./install.sh

This will:

- Configure environment variables
- Create CLI command: astroos

------------------------------------------------------------
Running Tests
------------------------------------------------------------

After activating environment:

    astroos unit_tests

Or:

    pytest src/test


------------------------------------------------------------
Python Version
------------------------------------------------------------

This project requires:

    Python 3.11



