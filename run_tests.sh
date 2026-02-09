
#!/usr/bin/env bash
set -euo pipefail

# Extend PYTHONPATH 
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q src/test 
