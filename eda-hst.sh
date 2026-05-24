
#make clean

rad init -n "3D-HST_Catalog_EDA"

#rad node -c -l "Exploratory analysis pipeline on the 3D-HST catalog"

# -------------------------------------------------
# Data import
# -------------------------------------------------
rad node -ct import -l "Import 3D-HST catalog"
rad node -i catalogs/collections/lsst-hst/hst/hst.fits
rad node -p max_records 300000
rad node -p script catalogs/collections/lsst-hst/hst/scripts/import.py

# -------------------------------------------------
# Data processing
# -------------------------------------------------
rad node -ct script -l "Clean"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/clean.py

rad node -ct script -l "Select"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/select.py

rad node -ct export -l "Export"

# -------------------------------------------------
# Exploratory data analysis
# -------------------------------------------------
rad node -ct eda-script -l "Distribution analysis"
rad node -p title "Exploratory distribution analysis"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/histogram_select.py
rad node -p eda_type histogram

rad node -ct eda-script -l "Pair-plot analysis"
rad node -p title "Exploratory pair-plot analysis"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/pair_plot_select.py
rad node -p eda_type pair-plot

rad node -ct eda-script -l "Sky distribution analysis"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/sky_distribution_select.py
rad node -p eda_type sky-distribution

#rad run

#cp "_pipelines/$(rad id)/dag.svg" ~/thesis-org/reports/paper/report/figures/dag_hst_eda/
#cp "_pipelines/$(rad id)/dag.yaml" ~/thesis-org/reports/paper/report/code/dag_hst_eda/

