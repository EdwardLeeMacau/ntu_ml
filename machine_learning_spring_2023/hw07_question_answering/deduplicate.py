"""
Find duplicated instance between train and dev set.

This script is generated by GitHub Copilot.
"""

import argparse
import json
from typing import List

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train", type=str, default="./cache/hw7_train.json", help="train question"
    )
    parser.add_argument(
        "--dev", type=str, default="./cache/hw7_dev.json", help="dev question"
    )

    return parser.parse_args()

"""
Train size: 26918
Dev size: 2863
Duplicated instance: 2256
"""
def count_duplicate(dataset):
    duplicated = []

    for i, x1 in enumerate(dataset['train']):
        for j, x2 in enumerate(dataset['dev']):
            if x1 == x2:
                duplicated.append((i, j))

    return duplicated

"""
Deduplicated instance: 607
"""
def deduplicate(dataset: List, duplicated: List):
    return [x for i, x in enumerate(dataset) if i not in duplicated]

def main():
    def load_dataset(path) -> List:
        with open(path, "r") as f:
            raw_dataset = json.load(f)

        return raw_dataset

    def instantiate(raw_dataset) -> List:
        return [
            (q["question"], q["context"], q["answers"]["text"]) for q in raw_dataset
        ]

    args = parse_args()
    dataset = {
        'train': load_dataset(args.train),
        'dev': load_dataset(args.dev)
    }
    instances = {
        'train': instantiate(dataset['train']),
        'dev': instantiate(dataset['dev'])
    }

    duplicated = [j for (_, j) in count_duplicate(instances)]
    dev = deduplicate(dataset['dev'], duplicated)
    print(f"Deduplicated instance: {len(dev)}")

    with open('./cache/hw7_dev_deduplicated.json', 'w', encoding='utf8') as f:
        json.dump(dev, f, ensure_ascii=False)

if __name__ == "__main__":
    main()