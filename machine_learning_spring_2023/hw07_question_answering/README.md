# How to run

1. Download dataset from competition. Assume it is stored under `./cache`.
2. Run `convert.sh` to convert the raw dataset to SQuAD formatted dataset.
3. (Optional) Deduplicate data in dev, using `deduplicate.py`. You can modify input directory with command line argument `--train <TRAIN_SET>` `--dev <DEV_SET>`. To modify output directory, you should modify the directory in code.
    ```python
    ...
    # Modify the directory of file write if need.
    with open('./cache/hw7_dev_deduplicated.json', 'w', encoding='utf8') as f:
        ...
    ```
4. Modify dataset directory and checkpoint saving directory in `train_bert.sh`

    ```bash
    $ vi train_bert.sh
    ```

    ```bash
    # Assume dataset is stored at /tmp2/dataset,
    # model checkpoints are stored at /tmp2/checkpoints
    ...
    python run_qa_no_trainer.py \
        ... \
        --train_file ./cache/hw7_train.json \
        --validation_file ./cache/hw7_dev.json \
        ... \
        --output_dir /tmp2/checkpoints \
        ...
    ```
5. The training script might save the checkpoint for every 200 steps. After updated 200 steps, stop the training script and use it to inference. Move it to root of checkpoint directory to apply the model parameter at inference time.
    ```bash
    /tmp2/checkpoints
    ├── config.json
    ├── eval_nbest_predictions.json
    ├── eval_predictions.json
    ├── pytorch_model.bin                <- overwrite this file by step_200/pytorch_model.bin
    ├── qa_no_trainer
    ├── special_tokens_map.json
    ├── step_200
    │   ├── optimizer.bin
    │   ├── pytorch_model.bin
    │   ├── random_states_0.pkl
    │   └── scheduler.bin
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
    ```
6. Create prediction file `submission.csv` using the following command.
    ```bash
    $ python predict.py --pad_to_max_length --question_file ./cache/hw7_test.json --predict_file submission.csv --model_name_or_path /tmp2/checkpoints --doc_stride 64
    ```


## Reference

1. My previous homework submission to [CSIE5431-Applied Deep Learning Fall 2022](https://github.com/EdwardLeeMacau/ntucsie_adl/tree/master/context_selection_and_question_answering).

    In that homework, we need to select relevant paragraph among 4 options, then extract the word from given context to answer the question. I adapt the script of question answering part as this submission. Notes that the script is adapt from [HuggingFace's sample code](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) originally.

    To pass the boss baseline, I found that using pretrained model `hfl/chinese-pert-large-mrc` is a MUST action. For other pretrained models, they produce $\leq 0.82$ accuracy.
