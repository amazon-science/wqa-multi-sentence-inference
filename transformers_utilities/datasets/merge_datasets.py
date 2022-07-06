from argparse import ArgumentParser
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


def main(args):
    assert all(os.path.isdir(filepath) for filepath in args.input), (
        "could not find some of the input datasets"
    )

    datas = [load_from_disk(filepath) for filepath in args.input]

    assert all(isinstance(d, Dataset) for d in datas) or all(isinstance(d, DatasetDict) for d in datas), (
        "datasets must be all DatasetDict or all Dataset"
    )

    if all(isinstance(d, DatasetDict) for d in datas):
        res = DatasetDict({
            split: concatenate_datasets([d[split] for d in datas], axis=1) for split in datas[0].keys()
        })
    else:
        res = concatenate_datasets(datas, axis=1)

    res.save_to_disk(args.output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, nargs='+', required=True, help="Input datasets to merge")
    parser.add_argument('--output', type=str, required=True, help="Output folder for resulting dataset")
    args = parser.parse_args()
    main(args)
