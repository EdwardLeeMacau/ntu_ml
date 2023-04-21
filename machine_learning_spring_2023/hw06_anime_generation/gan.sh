#!/bin/bash

stylegan2_pytorch --data ../../../dataset/crypko/faces \
    --image-size 64 \
    --batch-size 64 \
    --network-capacity 16 \
    --num-workers 8 \
    --num-train-steps 40000
