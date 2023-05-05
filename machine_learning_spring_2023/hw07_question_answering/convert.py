import argparse
import json
import os
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Fine tune a transformers model on a question answering task")
    parser.add_argument(
        "--raw-file", type=str, nargs='+', default=None, help="List of csv or json file containing the context data."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Where to store the converted dataset."
    )

    return parser.parse_args()


def to_squad_format(context: List[str], questions: List[Dict]) -> List[Dict]:
    """ Convert the context and questions to SQuAD format, to adapt the scripts from HuggingFace. """
    results = []
    for q in questions:
        q_formatted = {
            'id': str(q['id']),
            'question': q['question_text'],
            'title': '',
        }

        q_formatted['context'] = context[q['paragraph_id']]
        q_formatted['answers'] = {
            'answer_start': [q.get('answer_start', None)],
            'text': [q.get('answer_text', None)]
        }

        results.append(q_formatted)

    return results


def main():
    args = parse_args()

    for input_file in args.raw_file:
        with open(input_file, encoding='utf-8') as f:
            raw_data = json.load(f)

        context = raw_data['paragraphs']
        questions = raw_data['questions']

        # Create directory if it does not exist
        os.makedirs(args.output_dir, exist_ok=True)
        basename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, f"{basename}")

        if output_file == input_file:
            raise ValueError("Output file cannot be the same as input file.")

        # Wrap question in SQuAD format.
        squad: List = to_squad_format(context, questions)

        # Output file for question answering task
        with open(output_file, 'w') as f:
            json.dump(squad, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
