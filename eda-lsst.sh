
make clean
rad init-rsp

rad node-rsp -c -l "Exploratory analysis pipeline on the LSST DP-1 catalog"

rad node-rsp -ct tap -l "Query TAP service for LSST DP-1 catalog" --target extended_chandra_deep_field_south_ecdfs -m 1000 -d d1
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/query.py

rad node-rsp -ct script -l "Clean catalog"
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/clean.py

rad node-rsp -ct script -l "Select catalog"
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/select.py

rad node-rsp -ct export -l "Export processed catalog"

rad node-rsp -ct eda -l "Analyze catalog distributions"

rad node-rsp -ct eda-script -l "Catalog distribution analysis"
rad node-rsp -p title "Exploratory distribution analysis of the LSST DP-1 catalog"
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/histogram_select.py
rad node-rsp -p eda_type histogram

rad node-rsp -ct eda-script -l "Catalog sky distribution analysis"
rad node-rsp -p title "Exploratory sky distribution analysis of the LSST DP-1 catalog"
rad node-rsp -p script catalogs/collections/lsst-hst/lsst/scripts/sky_distribution_select.py
rad node-rsp -p eda_type sky-distribution

rad run-rsp


