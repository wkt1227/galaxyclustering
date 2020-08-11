#!/bin/bash
rm -r ./data/result/galaxycluster
mkdir data/result/galaxycluster

for i in {1..40}; do
    mkdir data/result/galaxycluster/$i
done