# How to run

## Environment

```bash
$ # Using Python 3.9
$ pip install -r requirements.txt
```

## Fine-tune the model

1. Convert the data to SQuAD format with script `convert.sh`
2. Fine-tune pretrained model from HuggingFace with script `train_qa.sh`.

## Generate prediction

1. Run `run.sh`

## Reference

1. My previous homework submission to [CSIE5431-Applied Deep Learning Fall 2022](https://github.com/EdwardLeeMacau/ntucsie_adl/tree/master/context_selection_and_question_answering).

    In that homework, we need to select relevant paragraph among 4 options, then answer the question from given paragraph. I adapt the script of question answering part as this submission. Notes that the script is adapt from [HuggingFace's sample code](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) originally.
