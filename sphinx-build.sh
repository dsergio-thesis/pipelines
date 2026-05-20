
sphinx-apidoc -f -o docs src/astroos_pipelines
cd docs && make clean && make html && cd -
