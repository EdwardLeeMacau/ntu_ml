# How to execute

1. Prepare fairseq scripts

    ```bash
    $ git clone https://github.com/pytorch/fairseq.git
    $ cd fairseq && git checkout 9a1c497
    $ pip install --upgrade ./fairseq/
    ```

2. Modify dataset directory and checkpoint saving directory in `params.yaml`.

    ```bash
    $ vi params.yaml
    ```

    ```yaml
    # Assume raw dataset is stored at /tmp2/ted2020
    # tokenize dataset is store at /tmp2/ted2020-bin
    # model checkpoints are stored at ./checkpoints/rnn
    env:
      dataset: /tmp2/ted2020
      binarized: /tmp2/ted2020-bin
      checkpoint: ./checkpoints/rnn
    ...
    ```

3. Execute `preprocess.py` to split the dataset into training, validation and test set. Then execute preprocessor implemented by fairseq to tokenize data.

    ```bash
    $ python preprocess.py
    $ python -m fairseq_cli.preprocess --source-lang en --target-lang zh --trainpref /tmp2/ted2020/train --validpref /tmp2/ted2020/valid --testpref /tmp2/ted2020/test --destdir /tmp2/ted2020-bin --joined-dictionary --workers 2
    ```

4. Execute `train.py` to train a model from stretch.

    ```bash
    $ python train.py
    ```

5. Average checkpoints

   ```bash
   $ python fairseq/scripts/average_checkpoints.py --inputs checkpoints/rnn --num-epoch-checkpoints 5 --output checkpoints/rnn/avg_last_5_checkpoint.pt
   ```

6. Execute `inference.py` with option `--load-ckpt <checkpoint>` to generate prediction.

    ```bash
    $ python inference.py --load-ckpt <checkpoint>
    ```
