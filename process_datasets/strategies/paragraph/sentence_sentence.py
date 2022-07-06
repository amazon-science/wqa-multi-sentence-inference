import random
from argparse import ArgumentParser, Namespace
from typing import Dict, Generator, List

from process_datasets.strategies.paragraph.base import _PairwiseStrategy
from process_datasets.utils.general import check_dict


class Sentence2SentenceStrategy(_PairwiseStrategy):
    r"""
    Sentence2SentenceStrategy introduces a slightly modified way to create examples. In Sentence2SentenceStrategy
    sentences are usually paired with some consecutive amount of text that makes sense.
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

        assert 0 < self.hparams.premise_range[0] <= self.hparams.premise_range[1], (
            "`--premise_range` must be a non-empty range with limits greater than 0"
        )
        assert 0 < self.hparams.consequence_range[0] <= self.hparams.consequence_range[1], (
            "`--consequence_range` must be a non-empty range with limits greater than 0"
        )

        self.hparams.premise_range = tuple(
            range(self.hparams.premise_range[0], self.hparams.premise_range[1] + 1)
        )
        self.hparams.consequence_range = tuple(
            range(self.hparams.consequence_range[0], self.hparams.consequence_range[1] + 1)
        )
        if self.hparams.premise_probs is None:
            self.hparams.premise_probs = (1.0, ) * len(self.hparams.premise_range)

        if self.hparams.consequence_probs is None:
            self.hparams.consequence_probs = (1.0, ) * len(self.hparams.consequence_range)

        assert len(self.hparams.premise_probs) == len(self.hparams.premise_range), (
            f"`premise_probs` must be of length {len(self.hparams.premise_range)}"
        )
        assert len(self.hparams.consequence_probs) == len(self.hparams.consequence_range), (
            f"`consequence_probs` must be of length {len(self.hparams.consequence_range)}"
        )
        assert self.hparams.max_hard_negatives is not None
        assert self.hparams.max_negatives is not None

    def get_random_premise_length(self) -> int:
        r""" Get random length of premise. """
        return random.choices(self.hparams.premise_range, weights=self.hparams.premise_probs, k=1)[0]

    def get_random_consequence_length(self) -> int:
        r""" Get random length of consequence. """
        return random.choices(self.hparams.consequence_range, weights=self.hparams.consequence_probs, k=1)[0]

    def create_examples(self) -> Generator[Dict, None, None]:
        r""" Create example with a single premise and multiple positive or negative consequences. """
        if check_dict(self.actual):
            combined = list(zip(self.actual['consequence'], self.actual['label']))
            random.shuffle(combined)
            self.actual['consequence'], self.actual['label'] = list(zip(*combined))
            yield self.actual
        self.actual = None

    def add_pair(self, premise: List[str], consequence: List[List[str]], label: int):
        r""" Add examples to a single instance. """
        premise = " ".join(premise)
        consequence = " ".join(consequence)

        if self.actual is None:
            self.actual = dict(premise=premise, consequence=[], label=[])
        assert premise == self.actual['premise'], (
            f"tried to add consequence of '{premise}' to '{self.actual['premise']}'"
        )

        self.actual['consequence'].append(consequence)
        self.actual['label'].append(label)

    def make(self, documents: Generator[List[List[str]], None, None]) -> Generator[Dict, None, None]:
        r""" Take a list of paragraph of the same document and create
        positive pairs from following sentences. A paragraph is itself a list of strings.
        """

        if self.hparams.selection == "following":
            consequence_on_the_right = True
        elif self.hparams.selection == "same":
            consequence_on_the_right = random.random() < 0.5

        documents = list(documents)

        document_indexes = list(range(len(documents)))

        choices = int(len(document_indexes) * self.hparams.document_ratio)
        if choices < len(document_indexes):
            document_indexes = random.sample(document_indexes, k=choices)

        for document_index in document_indexes:
            document = documents[document_index]

            # premise and consequence length
            premise_length = self.get_random_premise_length()
            positive_consequence_length = self.get_random_consequence_length()

            # select a paragraph that is able to contain both the premise and a consequence
            valid_indexes = [
                i for i, paragraph in enumerate(document)
                if len(paragraph) >= (premise_length + positive_consequence_length)
            ]

            if not valid_indexes:  # jump this document, very rare
                continue

            choices = max(1, int(self.hparams.paragraph_ratio * len(valid_indexes)))
            if choices < len(valid_indexes):
                valid_indexes = random.sample(valid_indexes, k=choices)

            for index in valid_indexes:

                # get paragraph
                selected_paragraph = document[index]  # list of sentences

                # premise position
                if consequence_on_the_right:
                    # going to take consequence after the premise
                    premise_position = random.randint(
                        0, len(selected_paragraph) - positive_consequence_length - premise_length
                    )

                    # positive consequence position
                    positive_consequence_position = random.randint(
                        premise_position + premise_length, len(selected_paragraph) - positive_consequence_length
                    )

                else:
                    # going to take consequence before the premise
                    premise_position = random.randint(
                        positive_consequence_length, len(selected_paragraph) - premise_length
                    )

                    # positive consequence position
                    positive_consequence_position = random.randint(
                        0, premise_position - positive_consequence_length
                    )

                # extract premise and consequence
                premise = selected_paragraph[premise_position:premise_position + premise_length]
                positive_consequence = selected_paragraph[
                    positive_consequence_position:positive_consequence_position + positive_consequence_length
                ]

                # create positive example
                self.add_pair(premise, positive_consequence, 1)

                # sample number of negatives and hard negatives
                max_negatives = self.get_random_max_negatives()
                max_hard_negatives = self.get_random_max_hard_negatives()

                # create hard negative pairs (up to max_hard_negatives)
                hard_negative_sentences = []

                if self.hparams.selection == "following":
                    # try to create negative from before premise in the same paragraph
                    # so only with `selection="following"`
                    super_hard_negative_length = self.get_random_consequence_length()
            
                    if super_hard_negative_length <= premise_position:
                        super_hard_negative_position = random.randint(0, premise_position - super_hard_negative_length)

                        # take consequence part before promise
                        hard_negative_sentences.append(selected_paragraph[
                            super_hard_negative_position:super_hard_negative_position + super_hard_negative_length
                        ])

                if len(document) > max_hard_negatives * 2:
                    hard_negative_paragraphs_indexes = set()
                    indexes = list(range(len(document)))
                    while len(hard_negative_paragraphs_indexes) < max_hard_negatives * 2:
                        new_paragraph_index = random.choice(indexes)
                        if new_paragraph_index != index:
                            hard_negative_paragraphs_indexes.add(new_paragraph_index)
                    other_paragraphs = [document[i] for i in hard_negative_paragraphs_indexes]

                else:
                    other_paragraphs = document[:index] + document[index + 1:]

                # hard negatives from other paragraphs of the same document
                for other_paragraph in other_paragraphs:

                    i = 0
                    while i < len(other_paragraph):
                        hard_negative_consequence_length = self.get_random_consequence_length()
                        hard_negative_sentences.append(other_paragraph[i:i + hard_negative_consequence_length])
                        i += hard_negative_consequence_length

                # if number of hard negatives is greater than limit, randomly sample from the list
                if len(hard_negative_sentences) > max_hard_negatives:
                    hard_negative_sentences = random.sample(hard_negative_sentences, k=max_hard_negatives)

                # create hard negatives
                for hard_negative_sentence in hard_negative_sentences:
                    self.add_pair(premise, hard_negative_sentence, 0)

                missing = max_negatives - len(hard_negative_sentences)
                while missing > 0:

                    other_document_index = document_index
                    while other_document_index == document_index:
                        other_document_index = random.randint(0, len(documents) - 1)

                    # get random other paragraph.
                    paragraph = random.choice(documents[other_document_index])

                    # paragraph may be too short
                    consequence_length = self.get_random_consequence_length()
                    if len(paragraph) >= consequence_length:

                        # get paragraph and create example
                        consequence_position = random.randint(0, len(paragraph) - consequence_length)
                        negative_consequence = paragraph[
                            consequence_position:consequence_position + consequence_length
                        ]

                        self.add_pair(premise, negative_consequence, 0)
                        missing -= 1

                yield from self.create_examples()

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        super(Sentence2SentenceStrategy, Sentence2SentenceStrategy).add_arguments_to_argparse(parser)
        parser.add_argument('--premise_range', required=True, type=int, nargs=2,
                            help="Min and max number of sentences in premise part.")
        parser.add_argument('--consequence_range', required=True, type=int, nargs=2,
                            help="Min and max number of sentences in consequence part.")
        parser.add_argument('--premise_probs', default=None, required=False, type=float, nargs='+',
                            help="Probabilities of sentences length in premise part.")
        parser.add_argument('--consequence_probs', default=None, required=False, type=float, nargs='+',
                            help="Probabilities of consequence length in premise part.")
        parser.add_argument('--selection', type=str, choices=('following', 'same'), default='following',
                            help="How to add for hard negatives.")
