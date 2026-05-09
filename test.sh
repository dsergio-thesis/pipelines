
make clean
rad init
rad node -c
rad node -ct import
rad node -i catalogs/hst/hst.fits
rad node -p max_records 1000
rad node -p script catalogs/hst/import.py
rad node -ct script
rad node -p script catalogs/hst/preprocess.py
rad node -ct export
rad node -ct eda
