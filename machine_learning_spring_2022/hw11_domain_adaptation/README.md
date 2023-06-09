# How to execute

1. Modify dataset directory and checkpoint saving directory in `params.yaml`.

    ```bash
    $ vi params.yaml
    ```

    ```yaml
    # Assume dataset is stored at /tmp2/real_or_drawing, best model is stored at /tmp2/real_or_drawing
    #
    # To avoid file overwriting, timestamp of script start time is used, the checkpoints are store
    # at /tmp2/real_or_drawing/<timestamp>/model_<iteration>.pt
    env:
      dataset: /tmp2/real_or_drawing
      checkpoint: /tmp2/real_or_drawing/<timestamp>/model_<iteration>.pt
    ...
    ```

2. Execute `main.py` with option `--train` to train a model from stretch.

    ```bash
    $ python main.py --train
    ```

3. Execute `main.py` with option `--test` and `--load-ckpt <checkpoint>` to generate prediction.

    ```bash
    $ python main.py --inference --load-ckpt <checkpoint>
    ```
