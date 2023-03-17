# Report

## Submit Prediction

```bash
$ kaggle competitions submit -c ml2023spring-hw2 -f <submission.csv> -m <message>
```

## Performance Matrix

| Commit ID |   Difference    | Metrics (Val) | Metrics (Dev) |
| :-------: | :-------------: | :-----------: | :-----------: |
|  09cbf00  | Epoch: 10 -> 30 |    0.51657    |       -       |
|     -     |       #1        |    0.68502    |    0.68802    |
|     -     |  BN, Epoch 100  |    0.65157    |               |

## Experiment

1. The following differences are applied:
   - Hidden dimension: 64 -> 256
   - Hidden layers: 2 -> 4
   - Input frames: 13 -> 17

## Reference

1. The average durations of phoneme vary from 74 ms to 181 ms.

## Report

1. (2%) Implement 2 models with approximately the same number of parameters, (A) one narrower and deeper (e.g. hidden_layers=6, hidden_dim=1024) and (B) the other wider and shallower (e.g.hidden_layers=2, hidden_dim=1750). Report training/validation accuracies for both models.

2. (2%) Add dropout layers, and report training/validation accuracies with
dropout rates equal to (A) 0.25/(B) 0.5/(C) 0.75 respectively
