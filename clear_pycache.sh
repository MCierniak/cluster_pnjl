#!/usr/bin/bash
for i in $(find ./src -name "__pycache__")
do
    rm -rf "$i"
done