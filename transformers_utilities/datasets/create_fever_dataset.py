import json
import logging
import os
from argparse import ArgumentParser
from typing import Dict
from datasets import Dataset, DatasetDict

from tqdm import tqdm


logging.getLogger().setLevel(logging.INFO)


label_to_id = {
    "SUPPORTS": 0,
    "NOT ENOUGH INFO": 1,
    "REFUTES": 2,
}

fake_evicence = ("no doc", -1, "", 0)


def get_dataset_from_file(filepath: str) -> Dict:
    r""" Build a single split of the dataset. """
    with open(filepath) as fi:
        data = [json.loads(line) for line in fi]

    res = dict(claim=[], evidence=[], label=[], key=[], doc=[])

    for example in tqdm(data, desc="Processing..."):
        label = example.get('label', None)
        if label is not None:
            label = label_to_id[label]

        if not example['evidence']:
            logging.warn(f"No evidence for claim id {example['id']}")
            example['evidence'] = [fake_evicence]

        evicences = [evidence[0] + ". " + evidence[2] for evidence in example['evidence']]
        doc_names = [evidence[0] for evidence in example['evidence']]

        res['claim'].append(example['claim'])
        res['evidence'].append(evicences)
        res['label'].append(label)
        res['doc'].append(doc_names)
        res['key'].append(example['id'])

    return Dataset.from_dict(res)


def main(args):
    r""" Create FEVER dataset. """

    assert os.path.isfile(args.train_file)
    assert os.path.isfile(args.dev_file)
    assert os.path.isfile(args.test_file)

    res = DatasetDict(
        train=get_dataset_from_file(args.train_file),
        validation=get_dataset_from_file(args.dev_file),
        test=get_dataset_from_file(args.test_file),
    )
    res.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()
    main(args)
