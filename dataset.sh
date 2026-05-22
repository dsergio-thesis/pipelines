
make clean
rad init-rsp

#rad node-rsp -c -l "LSST DP-1 and 3D-HST Catalog Dataset"

# -------------------------------------------------
# LSST DP-1 Data import
# -------------------------------------------------
rad node-rsp -ct tap -l "TAP LSST DP-1 catalog" --target extended_chandra_deep_field_south_ecdfs -m 5000 -d d1
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/query.py

# -------------------------------------------------
# LSST DP-1 Data processing
# -------------------------------------------------
rad node-rsp -ct script -l "Clean LSST DP-1"
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/clean.py

rad node-rsp -ct script -l "Select LSST DP-1"
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/select.py

rad node-rsp -ct export -l "Export LSST DP-1"

# -------------------------------------------------
# 3D-HST Data import
# -------------------------------------------------
rad node-rsp -cot import -l "Import 3D-HST"
rad node-rsp -i catalogs/collections/lsst-hst/hst/hst.fits
rad node-rsp -p max_records 300000
rad node-rsp -p script catalogs/collections/lsst-hst/hst/scripts/import.py

# -------------------------------------------------
# 3D-HST Data processing
# -------------------------------------------------
rad node-rsp -ct script -l "Clean 3D-HST"
rad node-rsp -p script catalogs/collections/lsst-hst/hst/scripts/clean.py

rad node-rsp -ct script -l "Select 3D-HST"
rad node-rsp -p script catalogs/collections/lsst-hst/hst/scripts/select.py

rad node-rsp -ct export -l "Export 3D-HST"

# -------------------------------------------------
# Data merging
# -------------------------------------------------
rad node-rsp -ct merge -l "Merge LSST DP-1 and HST"
rad node-rsp --parent "Export LSST DP-1"

rad node-rsp -ct export -l "Export merged dataset"

# -------------------------------------------------
# Construct Dataset
# -------------------------------------------------
rad node-rsp -ct photo-dataset -l "Construct LSST DP-1 and 3D-HST dataset"
rad node-rsp -p dataset-name "05-21-2"

rad node-rsp -ct "butler-coadd-cutout" -l "Generate cutouts for LSST DP-1 and 3D-HST dataset"
rad node-rsp -p dataset-name "05-21-2"

# -------------------------------------------------
# Exploratory data analysis
# -------------------------------------------------
rad node-rsp -ct eda-script -l "Catalog distribution analysis"
rad node-rsp -p title "Exploratory distribution analysis of LSST DP-1 and 3D-HST catalogs"
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/histogram_select.py
rad node-rsp -p eda_type histogram


#rad run-rsp

