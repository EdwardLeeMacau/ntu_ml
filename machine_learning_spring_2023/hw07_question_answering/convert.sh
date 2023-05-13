#!/bin/bash

# Before running this script, please download the dataset from competition
# ...

# Modify the file directory to your own
python convert.py \
    --raw-file /tmp2/edwardlee/dataset/drcd/hw7_train.json \
               /tmp2/edwardlee/dataset/drcd/hw7_dev.json \
               /tmp2/edwardlee/dataset/drcd/hw7_test.json \
    --output-dir cache
