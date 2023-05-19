# How to execute

1. Modify dataset directory and checkpoint saving directory in `params.yaml`.

    ```bash
    $ vi params.yaml
    ```

    ```yaml
    # Assume dataset is stored at /tmp2/anomaly, best model is stored at /tmp2/anomaly
    #
    # To avoid file overwriting, timestamp of script start time is used, the checkpoints are store
    # at /tmp2/anomaly/<timestamp>/model_<iteration>.pt
    env:
      dataset: /tmp2/anomaly
      checkpoint: /tmp2/anomaly/<timestamp>/model_<iteration>.pt
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
