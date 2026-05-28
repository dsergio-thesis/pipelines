
# Astroos LSST/3D-HST Machine Learning Pipelines 


### The project uses:

- Python 3.11
- pyproject.toml packaging
- Editable installs
- CLI interface via `rad`

------------------------------------------------------------

## Quick Start

Install:

    ./install.sh


------------------------------------------------------------
## CLI Usage
```
usage: rad <command> [options]

RAD - Provenance-Aware DAG Pipeline Manager
===========================================

init    -n, --name <pipeline-name>

node
  -c, --create                     Create node
  -o, --origin                     Mark origin node
  -l, --label <label>              Node label
  -t, --type <type>                import | export | script |
                                   eda-script | tap | gtap |
                                   photo-dataset |
                                   butler-coadd-cutout
  -i, --input-artifact <path>      Input artifact path
  -p, --parameter <key> <value>    Parameter assignment
  --parent <node-id>               Parent dependency

run                                Execute DAG pipeline
status                             Show pipeline status
checkout <pipeline-id>             Switch pipeline 
graph                              Render DAG visualization
export                             Export final artifacts
```
------------------------------------------------------------

# INSTALLATION GUIDE

From the project root (pipelines/):

    ./install.sh

This will:

- Configure environment variables
- Create CLI command: `rad`



------------------------------------------------------------
Python Version
------------------------------------------------------------

This project requires:

    Python 3.11
    Access to the RSP environment for LSST-specific operations (use `node-rsp` and `run-rsp`)



