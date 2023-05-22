#! /usr/bin/bash

python cpython_setup/setup_cbscs.py build
python cpython_setup/setup_cbscs.py clean
find ./build/ -name cbscs.* -exec mv {} "./src/pnjl/thermo/gcp_cluster/cbscs.so" ";"

rm -r ./build