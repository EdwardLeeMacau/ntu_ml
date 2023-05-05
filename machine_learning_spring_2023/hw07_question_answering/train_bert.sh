# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python run_qa_no_trainer.py \
  --model_name_or_path ckiplab/albert-base-chinese \
  --train_file ./cache/hw7_train.json \
  --validation_file ./cache/hw7_dev_deduplicated.json \
  --output_dir /tmp2/edwardlee/checkpoints/drcd/ckiplab-albert-base-chinese \
  --pad_to_max_length \
  --seed 0 \
  --with_tracking \
  --report_to tensorboard \
  --checkpointing_steps 200 \
  --learning_rate 1e-4 \
  --num_train_epochs 45 \
  --num_warmup_steps 5000 \
  --n_best_size 3 \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 64

