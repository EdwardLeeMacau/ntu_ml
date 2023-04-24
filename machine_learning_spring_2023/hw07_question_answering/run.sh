# "${1}": path to the fine-tuned model.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python predict.py \
    --pad_to_max_length \
    --question_answering_model $1 \
    --per_device_eval_batch_size 1 \
    --question_file $2 \
    --predict_file $3
