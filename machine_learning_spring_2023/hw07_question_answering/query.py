import json
import os
import csv
import pandas as pd
import argparse



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id", type=int, nargs='*', help="question id"
    )
    parser.add_argument(
        "--question", type=str, default="./cache/hw7_test.json", help="test question"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load questions
    with open(args.question, "r") as f:
        questions = json.load(f)

    # Load predictions for various models
    root = "reference"
    pred = {}
    for i, file in enumerate(os.listdir(root)):
        file = os.path.join(root, file)

        _, ext = os.path.splitext(file)
        if ext != ".csv":
            continue

        # Load csv
        df = pd.read_csv(file)
        df = df.drop([df.columns[0]], axis=1)

        pred[i] = df.values.flatten()

    pred = pd.DataFrame(pred)

    agreements = pred.eq(pred[0], axis=0).all(axis=1)

    agree_ratio = agreements.sum() / len(agreements)
    print(f'{agree_ratio=:.2%}')

    # Display disagreements
    for idx, value in agreements[agreements == False].items():
        q = questions[idx].copy()
        q['answers'] = pred.loc[idx].values.tolist()

        del q['id'], q['title']
        print(f'{q=}')

if __name__ == "__main__":
    main()