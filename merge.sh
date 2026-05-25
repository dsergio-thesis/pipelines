
#make clean
rad init -n "dag-merge"

rad node -cl "a1" -t import
rad node -i catalogs/collections/test_data/a1.csv
rad node -p script catalogs/collections/test_data/a1.py

rad node -cl "a2"

rad node -cl "a3" -t script
rad node -p script catalogs/collections/test_data/a3.py

rad node -col "b1" -t script
rad node --parent "a2"
rad node -p script catalogs/collections/test_data/b1.py

rad node -cl "b2" -t import
rad node -i catalogs/collections/test_data/b2.csv
rad node -p script catalogs/collections/test_data/b2.py

rad node -col "a4"
rad node --parent "a3"

rad node -cl "a5"
rad node --parent "b2"

rad node -ct export -l "a6"


rad run

