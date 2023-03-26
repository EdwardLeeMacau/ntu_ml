# How to execute

1. Modify dataset directory and checkpoint saving directory in `params.yaml`.

    ```bash
    $ vi params.yaml
    ```

    ```yaml
    # Assume dataset is stored at /tmp2/food-11, best model is stored at /tmp2/food-11-ckpt
    #
    # To avoid file overwriting, uuid1 is used, the checkpoints are store
    # at /tmp2/food-11-ckpt/<uuid1>/model_<iteration>.pt
    env:
      dataset: /tmp2/food-11
      checkpoint: /tmp2/food-11-ckpt
    ...
    ```

2. Execute `main.py` with option `--train` to train a model from stretch.

    ```bash
    $ python main.py --train
    ```

3. Execute `main.py` with option `--test` and `--load-ckpt <checkpoint>` to generate prediction.

    ```bash
    $ python main.py --test --load-ckpt <checkpoint>
    ```
