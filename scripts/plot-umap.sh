
# if env.sh does not exist, tell user to create it using env.sh.template
if [ ! -f env.sh ]; then
    echo "env.sh not found. Please create it using env.sh.template"
    exit 1
fi

source env.sh
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m clients.plot_umap "$@" 
