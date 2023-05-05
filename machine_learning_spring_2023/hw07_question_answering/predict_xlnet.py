import argparse
import csv
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions_with_beam_search
from typing import List, Dict
from datasets import Dataset
import pandas as pd
from transformers import (XLNetConfig, XLNetTokenizerFast, XLNetForQuestionAnswering)

import transformers
from transformers import (
    AdamW,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizerFast,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version


def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    """
    Save results while prefixing metric names.

    Args:
        results: (:obj:`dict`):
            A dictionary of results.
        output_dir: (:obj:`str`):
            An output directory.
        file_name: (:obj:`str`, `optional`, defaults to :obj:`all_results.json`):
            An output file name.
        metric_key_prefix: (:obj:`str`, `optional`, defaults to :obj:`eval`):
            A metric name prefix.
    """
    # Prefix all keys with metric_key_prefix + '_'
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question answering script, with BERT-based model.")
    parser.add_argument(
        "--question_file", type=str, required=True,
        help="A csv or a json file containing the prediction data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to question answering pre-trained model"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--predict_file", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Run with CPU only mode."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )

    return parser.parse_args()


# Returns predictions
# Can evaluate model accuracy (EM) and f1-score
def question_answering(questions: List[Dict], args: argparse.Namespace) -> List[str]:
    accelerator = Accelerator()

    # Prepare dataset
    # When using your own dataset or a difference dataset from swag, you will probably need to change this.
    raw_datasets = Dataset.from_dict(
        pd.DataFrame(questions).to_dict(orient='list')
    )

    config = XLNetConfig.from_pretrained(args.model_name_or_path)
    tokenizer = XLNetTokenizerFast.from_pretrained(args.model_name_or_path)
    model = XLNetForQuestionAnswering.from_pretrained(
        args.model_name_or_path, from_tf=False, config=config
    )

    column_names = raw_datasets.column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_length, tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
        special_tokens = tokenized_examples.pop("special_tokens_mask")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        # We still provide the index of the CLS token and the p_mask to the model, but not the is_impossible label.
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []

        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # Find the CLS token in the input ids.
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples["cls_index"].append(cls_index)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples["token_type_ids"][i]
            for k, s in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            context_idx = 1 if pad_on_right else 0

            # Build the p_mask: non special tokens and context gets 0.0, the others 1.0.
            tokenized_examples["p_mask"].append(
                [
                    0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                    for k, s in enumerate(sequence_ids)
                ]
            )

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_idx else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    with accelerator.main_process_first():
        dataset = raw_datasets.map(
            prepare_validation_features,
            batched=True,
            remove_columns=raw_datasets.column_names
        )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    dataset_for_model = dataset.remove_columns(["example_id", "offset_mapping"])
    dataloader = DataLoader(
        dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, scores_diff_json = postprocess_qa_predictions_with_beam_search(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            start_n_top=model.config.start_n_top,
            end_n_top=model.config.end_n_top,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": scores_diff_json[k]}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    all_start_top_log_probs = []
    all_start_top_index = []
    all_end_top_log_probs = []
    all_end_top_index = []
    all_cls_logits = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            outputs = model(**batch)
            start_top_log_probs = outputs.start_top_log_probs
            start_top_index = outputs.start_top_index
            end_top_log_probs = outputs.end_top_log_probs
            end_top_index = outputs.end_top_index
            cls_logits = outputs.cls_logits

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_top_log_probs = accelerator.pad_across_processes(start_top_log_probs, dim=1, pad_index=-100)
                start_top_index = accelerator.pad_across_processes(start_top_index, dim=1, pad_index=-100)
                end_top_log_probs = accelerator.pad_across_processes(end_top_log_probs, dim=1, pad_index=-100)
                end_top_index = accelerator.pad_across_processes(end_top_index, dim=1, pad_index=-100)
                cls_logits = accelerator.pad_across_processes(cls_logits, dim=1, pad_index=-100)

            all_start_top_log_probs.append(accelerator.gather_for_metrics(start_top_log_probs).cpu().numpy())
            all_start_top_index.append(accelerator.gather_for_metrics(start_top_index).cpu().numpy())
            all_end_top_log_probs.append(accelerator.gather_for_metrics(end_top_log_probs).cpu().numpy())
            all_end_top_index.append(accelerator.gather_for_metrics(end_top_index).cpu().numpy())
            all_cls_logits.append(accelerator.gather_for_metrics(cls_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_end_top_log_probs])  # Get the max_length of the tensor

    # concatenate all numpy arrays collected above
    start_top_log_probs_concat = create_and_fill_np_array(all_start_top_log_probs, dataset, max_len)
    start_top_index_concat = create_and_fill_np_array(all_start_top_index, dataset, max_len)
    end_top_log_probs_concat = create_and_fill_np_array(all_end_top_log_probs, dataset, max_len)
    end_top_index_concat = create_and_fill_np_array(all_end_top_index, dataset, max_len)
    cls_logits_concat = np.concatenate(all_cls_logits, axis=0)

    outputs_numpy = (
        start_top_log_probs_concat,
        start_top_index_concat,
        end_top_log_probs_concat,
        end_top_index_concat,
        cls_logits_concat,
    )

    prediction = post_processing_function(raw_datasets, dataset, outputs_numpy)

    return prediction


def main():
    args = parse_args()

    with open(args.question_file, encoding='utf-8') as f:
        questions = json.load(f)

    if args.debug:
        args.cpu = True
        questions = questions[:10]

    answers = question_answering(questions, args).predictions
    with open(args.predict_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])

        for question, answer in zip(questions, answers):
            assert(question['id'] == answer['id'])
            writer.writerow([question['id'], answer['prediction_text']])


if __name__ == "__main__":
    main()
