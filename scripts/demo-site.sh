
# if env.sh does not exist, tell user to create it using env.sh.template
if [ ! -f env.sh ]; then
    echo "env.sh not found. Please create it using env.sh.template"
    exit 1
fi

source env.sh
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python -m clients.demo_site "$@" 

rm -rf ~/thesis-org/astroml.github.io/site/data
rm -rf ~/thesis-org/astroml.github.io/site/assets/plots/*.png
cp -r ~/thesis-org/pipelines/site/data ~/thesis-org/astroml.github.io/site/data
cp -r ~/thesis-org/pipelines/site/assets/* ~/thesis-org/astroml.github.io/site/assets


