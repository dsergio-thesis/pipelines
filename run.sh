

# if env.sh does not exist, tell user to create it using env.sh.template
if [ ! -f env.sh ]; then
    echo "env.sh not found. Please create it using env.sh.template"
    exit 1
fi

source env.sh

PYTHONPATH=src python -m test.client_lsst_pipeline "$@"
