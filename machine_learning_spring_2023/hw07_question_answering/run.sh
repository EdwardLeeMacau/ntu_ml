# "${1}": path to the fine-tuned model.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python predict_bert.py \
    --pad_to_max_length \
    --question_answering_model $1 \
    --per_device_eval_batch_size 16 \
    --doc_stride 64 \
    --max_answer_length 64 \
    --question_file $2 \
    --predict_file $3
