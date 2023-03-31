# Report

## Submit Prediction

```bash
$ kaggle competitions submit -c ml2023spring-hw3 -f <submission.csv> -m <message>
```

## Applied Hyperparameters

```
epochs: 768
optimizer:
  name: RAdam
  kwargs:
    lr: 1.0e-4
    weight_decay: 1.0e-4
scheduler:
  name: StepLR
  kwargs:
    gamma: 0.1
    step_size: 512
    step_unit: epoch
batch-size: 64
mixup:
  enable: true
  alpha: 0.4
test-time-augmentation:
  weight: 0.2
  candidates: 11
seed: 3407
early-stop: 1024
input-size: 224
```

### Ensemble Setup

model 1 (`resnext101_32_8d`):
- Use random seed 6666 and update model with 112098 iterations.
- Do not shuffle training set and validation set in this trail, just comment out L123 to L131 in main.py

model 2 (`resnext101_32_8d`):
- Use random seed 3407 and update model with 115238 iterations.
- Apply training set and validation set shuffling

model 3 (`resnext101_32_8d`):
- Use random seed 105 and update model with 120419 iterations.
- Apply training set and validation set shuffling

## Report

1. Augmentation Implementation

2. Visual Representations Implementation