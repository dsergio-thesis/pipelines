mkdir -p uml
pyreverse -ASmy -o png -p astroos_pipelines src/astroos_pipelines
mv *.png uml/
