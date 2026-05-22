
make clean
rad init

rad node -c -l "Galaxy Zoo Dataset"

# -------------------------------------------------
# Data import
# -------------------------------------------------
rad node -ct gtap -l "TAP Query"
rad node -p max_records 300
rad node -p script catalogs/collections/gzoo/scripts/query.py
