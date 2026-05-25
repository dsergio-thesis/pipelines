
# Astroos LSST/SDSS/HST Machine Learning Pipelines 

## Overview

- Dataset generation
- LSST
- SDSS
- DESI Legacy Survey
- Galaxy Zoo
- Survey photometry/pixel pipelines

### The project uses:

- Python 3.11
- pyproject.toml packaging
- Editable installs
- CLI interface via `rad`

------------------------------------------------------------

## Quick Start

Install:

    ./install.sh

Run tests:

    rad unit_tests

------------------------------------------------------------
## CLI Usage

The installer creates a CLI command:

    rad <command> [args...]

Commands are auto-detected from:

    $ASTROOS_ROOT/scripts/

Examples:

    rad unit_tests

------------------------------------------------------------

# INSTALLATION GUIDE

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


------------------------------------------------------------
Python Version
------------------------------------------------------------

This project requires:

    Python 3.11



