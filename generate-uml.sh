mkdir -p uml

pyreverse \
    -A \
    --ignore=hst,logger,mast,sdss \
    -o png \
    -p radstroos \
    src/astroos_pipelines/

mv *.png uml/
