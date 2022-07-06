import logging
import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple
import csv

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm


logging.getLogger().setLevel(logging.INFO)


def load_filter(filepath: str) -> List[str]:
    r""" Load a linewise file into an array of strings. """
    with open(filepath) as fi:
        return set(line.strip() for line in fi)


def load_dataset(filepath: str) -> List[Tuple]:
    r""" Load ASNQ dataset in TSV from disk. """
    with open(filepath) as fi:
        reader = csv.reader(fi, delimiter='\t', quoting=csv.QUOTE_NONE)
        yield from reader


def process_split(dataset: List[Tuple], question_filter: List[str] = None) -> Dict:
    r""" Process a single split and filter on questions that are in filter. """
    questions_to_answers = {}
    for question, candidate, label in tqdm(dataset, desc=f"Processing..."):
        if question not in questions_to_answers:
            questions_to_answers[question] = {
                'answer': [],
                'label': [],
                'key': len(questions_to_answers)
            }

        questions_to_answers[question]['answer'].append(candidate)
        questions_to_answers[question]['label'].append(int(label.strip() == '4'))

    if question_filter is not None:
        questions_to_answers = {
            k: v for k, v in questions_to_answers.items() if k in question_filter
        }

    dataset = dict(question=[], answer=[], label=[], key=[])
    for question in sorted(list(questions_to_answers.keys())):
        values = questions_to_answers[question]
        dataset['question'].append(question)
        dataset['answer'].append(values['answer'])
        dataset['key'].append(values['key'])
        dataset['label'].append(values['label'])

    return Dataset.from_dict(dataset)


def main(args):
    r""" Create ASNQ dataset. """
    assert not os.path.exists(args.output_folder)

    logging.info("Loading data")
    asnq = {
        'train': load_dataset(os.path.join(args.input_folder, 'train.tsv')),
        'validation': load_dataset(os.path.join(args.input_folder, 'dev.tsv')),
        'test': load_dataset(os.path.join(args.input_folder, 'dev.tsv'))
    }

    logging.info("Loading filters")
    filters = {
        'dev': load_filter(args.dev_filter),
        'test': load_filter(args.test_filter)
    }

    res = DatasetDict(
        train=process_split(asnq['train']),
        validation=process_split(asnq['validation'], question_filter=filters['dev']),
        test=process_split(asnq['test'], question_filter=filters['test']),
    )

    logging.info("Saving results")
    res.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--dev_filter', type=str, required=True)
    parser.add_argument('--test_filter', type=str, required=True)
    args = parser.parse_args()
    main(args)
