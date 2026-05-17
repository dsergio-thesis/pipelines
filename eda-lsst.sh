
make clean
rad init

rad node -c -l "Exploratory analysis pipeline on the LSST DP-1 catalog"

rad node-rsp -ct tap -l "Query TAP service for LSST DP-1 catalog" --target extended_chandra_deep_field_south_ecdfs -m 300 -d d1
rad node -p script catalogs/collections/lsst-hst/lsst/scripts/query.py

rad node -ct script -l "Clean catalog"
rad node -p script catalogs/collections/lsst-hst/lsst/scripts/clean.py

rad node -ct script -l "Select catalog"
rad node -p script catalogs/collections/lsst-hst/lsst/scripts/select.py

rad node -ct export -l "Export processed catalog"

rad node -ct eda -l "Analyze catalog distributions"
rad node -p title "Exploratory analysis of the LSST DP-1 catalog"

rad run

cp "_pipelines/$(rad id)/dag.svg" ~/thesis-org/reports/paper/report/figures/dag_lsst_eda/
cp "_pipelines/$(rad id)/dag.yaml" ~/thesis-org/reports/paper/report/code/dag_lsst_eda/

