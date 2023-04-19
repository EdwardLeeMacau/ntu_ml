#!/bin/bash

stylegan2_pytorch --data ../../../dataset/crypko/faces \
    --image-size 64 \
    --network-capacity 64 \
    --generate \
    --num-image-tiles 1 \
    --num-generate 1000
