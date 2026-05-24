
#make clean
rad init

# -------------------------------------------------
# Data import
# -------------------------------------------------
rad node -ct gtap -l "TAP Query - GZOO"
rad node -p max_records 10000
rad node -p script catalogs/collections/gzoo/scripts/query_gz_tr.py
rad node -p base_url "https://datalab.noirlab.edu/tap"

rad node -ct export -l "Export GZ-Tractor"

# -------------------------------------------------
# Data processing
# -------------------------------------------------
rad node -ct script -l "Label Dataset"
rad node -p script catalogs/collections/gzoo/scripts/label.py
rad node -p labels catalogs/collections/gzoo/morphology_labels.csv

# -------------------------------------------------
# Data export and EDA
# -------------------------------------------------
#rad node -ct export -l "Export FITS"

rad node -ct eda-script -l "Catalog distribution analysis"
rad node -p title "Exploratory distribution analysis of the Galaxy Zoo catalog"
rad node -p script catalogs/collections/gzoo/scripts/histogram_select.py
rad node -p eda_type histogram


# -------------------------------------------------
# Fetch Cutout Images
# -------------------------------------------------
rad node -ct ls-cutout -l "Fetch Cutouts"
rad node -p dataset-name "05-23-10" --labels-def-file "catalogs/collections/gzoo/morphology_labels.csv"




