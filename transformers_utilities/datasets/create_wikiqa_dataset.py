import logging
import os
from argparse import ArgumentParser

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm


logging.getLogger().setLevel(logging.INFO)


def main(args):
    r""" Create WikiQA dataset. """
    assert not os.path.exists(args.output_folder)

    logging.info("Loading data")
    wikiqa = load_dataset('wiki_qa')

    res = {}
    for split in wikiqa.keys():
        dataset = wikiqa[split]

        questions_to_answers = {}
        for example in tqdm(dataset, total=len(dataset), desc=f"Processing split {split}..."):
            if example['question'] not in questions_to_answers:
                questions_to_answers[example['question']] = {
                    'answer': [],
                    'label': [],
                    'key': example['question_id']
                }

            questions_to_answers[example['question']]['answer'].append(example['answer'])
            questions_to_answers[example['question']]['label'].append(example['label'])

        if split in ('validation', 'test'):  # cleaning all+ and all-
            questions_to_answers = {
                k: v for k, v in questions_to_answers.items() if (0 < sum(v['label']) < len(v['label']))
            }

        dataset = dict(question=[], answer=[], label=[], key=[])
        for question in sorted(list(questions_to_answers.keys())):
            values = questions_to_answers[question]
            dataset['question'].append(question)
            dataset['answer'].append(values['answer'])
            dataset['key'].append(values['key'])
            dataset['label'].append(values['label'])

        res[split] = Dataset.from_dict(dataset)

    logging.info("Saving results")
    res = DatasetDict(res)
    res.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()
    main(args)
