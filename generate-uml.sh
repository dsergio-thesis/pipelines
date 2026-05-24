mkdir -p uml

#pyreverse \
    #-A \
    #--ignore=hst,logger,mast,sdss \
    #-o png \
    #-p dag \
    #src/astroos_pipelines/dag.py

#pyreverse \
    #-Ak \
    #-o png \
    #-p artifacts \
    #src/astroos_pipelines/artifacts.py

#pyreverse \
    #-A \
    #--ignore=hst,logger,mast,sdss \
    #-o png \
    #-p datasets \
    #src/astroos_pipelines/datasets.py

pyreverse \
    -A \
    -o png \
    -p morphometry \
    src/astroos_pipelines/morphometry.py


mv *.png uml/
