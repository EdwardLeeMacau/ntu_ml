# Report

## Reference

1. The average durations of phoneme vary from 74 ms to 181 ms.
See "Section 5.2 Phonemes Duration in Correlation
with the Position in the Sentence." in [Length of Phonemes in a Context of their Positions in Polish Sentences](https://www.scitepress.org/papers/2013/45035/45035.pdf).

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
   n-frames: 15
   ```

1. (2%) Implement 2 models with approximately the same number of parameters, (A) one narrower and deeper (e.g. hidden_layers=6, hidden_dim=1024) and (B) the other wider and shallower (e.g.hidden_layers=2, hidden_dim=1750). Report training/validation accuracies for both models.

   In this experiment, the dropout probability is set as 0.5.

   |        Model        | No. of RNN layers | Hidden Dim | Trainable Params | Train Acc | Val Acc |
   | :-----------------: | :---------------: | :--------: | :--------------: | :-------: | :-----: |
   | Narrower and deeper |        10         |    256     |     11960361     |  0.9788   | 0.8220  |
   | Wider and shallower |         3         |    512     |     12867625     |  0.9955   | 0.8112  |

   Learning curve is shown below

   - In training set, shallow model performs better than deep model. ![](q1.train.png)
   - In validation set, deep model performs better than shallow model. ![](q1.validation.png)

2. (2%) Add dropout layers, and report training/validation accuracies with dropout rates equal to (A) 0.25/(B) 0.5/(C) 0.75 respectively

   In this experiment, I use the predefined dropout layers in RNN.

   | Dropout | Train Acc | Val Acc |
   | :-----: | :-------: | :-----: |
   |  0.25   |  0.9952   | 0.8215  |
   |  0.50   |  0.9788   | 0.8220  |
   |  0.75   |  0.9404   | 0.7780  |

   - In training set, model with lower dropout rate has higher accuracy. ![](accuracy.train.png)
   - In validation set, model with dropout rate 0.75 has worse performance than others. ![](accuracy.validation.png)

