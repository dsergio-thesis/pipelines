mkdir -p uml

pyreverse \
    -A \
    --ignore=hst,logger,mast,sdss \
    -o png \
    -p dag \
    src/astroos_pipelines/dag.py

pyreverse \
    -Ak \
    --ignore=hst,logger,mast,sdss \
    -o png \
    -p artifacts \
    src/astroos_pipelines/artifacts.py

mv *.png uml/
