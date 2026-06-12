
conda activate astroos-pipelines-py311

mkdir -p demo

echo "A,B,C\n10,1.1,True\n20,1.2,False" > demo/a1.csv
cat demo/a1.csv

echo "active_columns.update({'A': 'A', 'B': 'B'})" > demo/a1.py
cat demo/a1.py

rad init -n "demo"

rad node -cl "a1" -t import
rad node -i demo/a1.csv
rad node -p script demo/a1.py

rad node -cl "a2"

echo "df['A'] = df['A'] * 2" > demo/a3.py
cat demo/a3.py

rad node -cl "a3: branch" -t script
rad node -p script demo/a3.py

echo "df['B'] = df['B'] * 2 # no merge conflict" > demo/b1.py
cat demo/b1.py

rad node -col "b1: branch" -t script
rad node --parent "a2"
rad node -p script demo/b1.py

echo "D,E\n100,1\n200,2" > demo/b2.csv
cat demo/b2.csv

echo "active_columns.update({'D': 'D', 'E': 'E'})" > demo/b2.py
cat demo/b2.py

rad node -cl "b2" -t import
rad node -i demo/b2.csv
rad node -p script demo/b2.py

rad node -col "a4"
rad node --parent "a3: branch"

rad node -cl "a5: merge"
rad node --parent "b2"

rad node -ct export -l "a6"

rad run

cat demo/b1.py

echo "df['A'] = df['A'] * 2 # merge conflict, but same hash" > demo/b1.py

cat demo/b1.py

rad run

echo "df['A'] = df['A'] * 3 # merge conflict, different hash" > demo/b1.py

cat demo/b1.py

rad run
