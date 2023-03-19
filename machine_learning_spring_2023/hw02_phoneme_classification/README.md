# Report

## Submit Prediction

```bash
$ kaggle competitions submit -c ml2023spring-hw2 -f <submission.csv> -m <message>
```

## Reference

1. The average durations of phoneme vary from 74 ms to 181 ms.

## Report

0. In the following questions, I use the following model as the experiment target.

   ```
   SeqTagger(
      (rnn): GRU(585, 256, num_layers=10, batch_first=True, dropout=dropout, bidirectional=True)
      (activation): Tanh()
      (classifier): Linear(in_features=512, out_features=41, bias=True)
   )
   ```

   Here are hyperparameters I used

   ```yaml
   epochs: 250
   optimizer:
      name: RAdam
      kwargs:
         lr: 3.0e-4
         weight_decay: 1.0e-5
   batch-size: 32
   seed: 3407
   train-val-ratio: 0.99
   scheduler:
      name: StepLR
      kwargs:
         gamma: 0.1
         step_size: 200
         step_unit: epoch
   model-input: concatenated 15-frames
   ```

1. (2%) Implement 2 models with approximately the same number of parameters, (A) one narrower and deeper (e.g. hidden_layers=6, hidden_dim=1024) and (B) the other wider and shallower (e.g.hidden_layers=2, hidden_dim=1750). Report training/validation accuracies for both models.

   In this experiment, the dropout probability is set as 0.5.

   - Narrower and deeper model (rnn.layers=10, hidden.dim=256)
      - trainable_param=11960361
   - Wider and shallower model (rnn.layers=3, hidden.dim=512)
      - trainable_param=12867625

   Learning curve is shown below

   - In training set, shallow model performs better than deep model. ![](q1.train.png)
   - In validation set, deep model performs better than shallow model. ![](q1.validation.png)

2. (2%) Add dropout layers, and report training/validation accuracies with dropout rates equal to (A) 0.25/(B) 0.5/(C) 0.75 respectively

   In this experiment, I use the predefined dropout layers in RNN.

   ![](accuracy.train.png)
   - In validation set, model with high dropout rate has worse performance than others. ![](accuracy.validation.png)

   | Dropout | Validation Accuracy |
   | :-----: | :-----------------: |
   |  0.25   |       0.82625       |
   |  0.50   |       0.82592       |
   |  0.75   |       0.78513       |
