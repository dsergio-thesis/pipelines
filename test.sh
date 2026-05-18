
make clean
rad init

rad node -cl "n1"
rad node -cl "n2"

rad node -col "n3"
rad node -cl "n4"

rad node -cl "n5" 
rad node --parent "n2"

rad node -cl "n6"

#rad run

