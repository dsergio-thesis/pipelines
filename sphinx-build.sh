
sphinx-apidoc -f -o docs \
    src/astroos_pipelines \
    "src/astroos_pipelines/sdss" \
    "src/astroos_pipelines/hst" \
    "src/astroos_pipelines/mast" 
cd docs && make clean && make html && cd -
