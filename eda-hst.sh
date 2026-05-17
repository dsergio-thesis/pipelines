
make clean
rad init

rad node -c -l "Exploratory analysis pipeline on the 3D-HST catalog"

rad node -ct import -l "Import catalog"
rad node -i catalogs/collections/lsst-hst/hst/hst.fits
rad node -p max_records 300000
rad node -p script catalogs/collections/lsst-hst/hst/scripts/import.py

#rad run

rad node -ct script -l "Clean catalog"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/clean.py

rad node -ct script -l "Select catalog"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/select.py

rad node -ct export -l "Export processed catalog"

#rad node -ct eda -l "Analyze catalog distributions"

rad node -ct eda-script -l "Catalog distribution analysis"
rad node -p title "Exploratory distribution analysis of the 3D-HST catalog"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/histogram_select.py
rad node -p eda_type histogram

rad node -ct eda-script -l "Catalog sky distribution analysis"
rad node -p title "Exploratory sky distribution analysis of the 3D-HST catalog"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/sky_distribution_select.py
rad node -p eda_type sky-distribution

rad run

#cp "_pipelines/$(rad id)/dag.svg" ~/thesis-org/reports/paper/report/figures/dag_hst_eda/
#cp "_pipelines/$(rad id)/dag.yaml" ~/thesis-org/reports/paper/report/code/dag_hst_eda/

