import torch
import random
import json
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# To clean model output. If you try different prompts, you may have to fix
# this function on your own
def clean_text(text):
    # Note: When you use unilingual model, the colon may become fullwidth
    text = text.split("答案:")[-1]
    text = text.split(" ")[0]
    return text

@torch.no_grad()
def main():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    root_dir = "/tmp2/edwardlee/dataset/drcd"

    same_seeds(2)

    # You can try model with different size
    # When using Colab or Kaggle, models with more than 2 billions parameters may
    # run out of memory
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-1.7B")
    model = AutoModelForCausalLM.from_pretrained("facebook/xglm-1.7B")

    # To avoid CUDA_OUT_OF_MEMORY
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    with open(os.path.join(root_dir, "hw7_in-context-learning-examples.json"), "r") as f:
        test = json.load(f)

    # K-shot learning
    # Give model K examples to make it achieve better accuracy
    # Note: (1) When K >= 4, CUDA_OUT_OFF_MEMORY may occur.
    #       (2) The maximum input length of XGLM is 2048
    K = 4

    question_ids = [qa["id"] for qa in test["questions"]]

    with open("in-context-learning-result.txt", "w") as f:
        print("ID,Ground-Truth,Prediction", file = f)
        for idx, qa in enumerate(test["questions"]):
            # You can try different prompts
            prompt = "請從最後一篇的文章中找出最後一個問題的答案，將答案覆述一次\n"
            # prompt = ""
            exist_question_indexs = [question_ids.index(qa["id"])]

            # K-shot learning: give the model K examples with answers
            for i in range(K):
                question_index = question_ids.index(qa["id"])
                while(question_index in exist_question_indexs):
                    question_index = random.randint(0, len(question_ids) - 1)
                exist_question_indexs.append(question_index)
                paragraph_id = test["questions"][question_index]["paragraph_id"]
                prompt += f'文章：{test["paragraphs"][paragraph_id]}\n'
                prompt += f'問題：{test["questions"][question_index]["question_text"]}\n'
                prompt += f'答案：{test["questions"][question_index]["answer_text"]}\n'

            # The final one question without answer
            paragraph_id = qa["paragraph_id"]
            prompt += f'文章：{test["paragraphs"][paragraph_id]}\n'
            prompt += f'問題：{qa["question_text"]}\n'
            prompt += f'答案：'

            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            sample = model.generate(**inputs, max_new_tokens = 20)
            text = tokenizer.decode(sample[0], skip_special_tokens=True)

            # Note: You can delete this line to see what will happen
            text = clean_text(text)

            print(prompt)
            print(f'正確答案: {qa["answer_text"]}')
            print(f'模型輸出: {text}')
            print()

            print(f"{idx},{qa['answer_text']},{text}", file = f)

if __name__ == "__main__":
    main()
