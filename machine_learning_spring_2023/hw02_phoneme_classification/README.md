# How to execute

1. Modify dataset directory and checkpoint saving directory in `params.yaml`.

    ```bash
    $ vi params.yaml
    ```

    ```yaml
    # Assume dataset is stored at /tmp2/libriphone, best model is stored at /tmp2/libriphone-ckpt
    env:
    dataset: /tmp2/libriphone
    checkpoint: /tmp2/libriphone-ckpt
    ...
    ```

2. Execute `main.py` with option `--train` to train a model from stretch.

    ```bash
    $ python main.py --train
    ```

3. Execute `main.py` with option `--test` to generate prediction.

    ```bash
    $ python main.py --test
    ```
