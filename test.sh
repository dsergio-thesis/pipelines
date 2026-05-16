
make clean
rad init

rad node -c -l "HST data processing pipeline"

rad node -ct import -l "Import HST data from FITS file"
rad node -i catalogs/collections/lsst-hst/hst/hst.fits
rad node -p max_records 1000
rad node -p script catalogs/collections/lsst-hst/hst/scripts/import.py

rad node -ct script -l "Clean HST data"
rad node -p script catalogs/collections/lsst-hst/hst/scripts/clean.py

rad node -ct export -l "Export HST data to FITS file"
#rad node -ct eda

rad run
