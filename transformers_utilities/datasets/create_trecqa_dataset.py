import json
import logging
import os
import gzip
from argparse import ArgumentParser
from typing import Dict, List

from datasets import Dataset, DatasetDict


logging.getLogger().setLevel(logging.INFO)


def to_bool(string):
    return string.lower() in ('yes', 'pos', 'positive', '1', 'correct')


def to_int(string):
    return int(to_bool(string))


def load_dataset(filepath: str) -> List[Dict]:
    r""" Load a JSONL from disk. """
    with (gzip.open(filepath) if filepath.endswith('gz') else open(filepath)) as fi:
        return [json.loads(line) for line in fi]


def main(args):
    r""" Create TREC-QA dataset. """
    assert not os.path.exists(args.output_folder)

    logging.info("Loading data")
    trecqa = {
        'train': load_dataset(os.path.join(args.input_folder, 'train-all.jsonl.gz')),
        'validation': load_dataset(os.path.join(args.input_folder, 'dev-filtered.jsonl')),
        'test': load_dataset(os.path.join(args.input_folder, 'test-filtered.jsonl')),
    }

    res = {}
    for split, data in trecqa.items():
        dataset = dict(question=[], answer=[], label=[], key=[])
        for i, sample in enumerate(sorted(data, key=lambda a: a['question'])):
            labels = [s['label'] for s in sample["candidates"]]
            answers = [s['sentence'] for s in sample["candidates"]]
            dataset['question'].append(sample['question'])
            dataset['answer'].append(answers)
            dataset['key'].append(i)
            dataset['label'].append(labels)

        res[split] = Dataset.from_dict(dataset)

    logging.info("Saving results")
    res = DatasetDict(res)
    res.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()
    main(args)
