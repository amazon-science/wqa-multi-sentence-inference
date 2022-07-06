import random
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Dict, Generator, List, Tuple

from process_datasets.strategies.strategy import Strategy
from process_datasets.utils.general import clean_documents


class _ParagraphStrategy(Strategy):

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

        assert self.hparams.min_sentence_length >= 1, (
            "`--min_sentence_length` must be a positive integer"
        )
        assert self.hparams.min_paragraph_length >= 1, (
            "`--min_paragraph_length` must be a positive integer"
        )
        assert self.hparams.min_document_length >= 1, (
            "`--min_document_length` must be a positive integer"
        )

        assert self.hparams.min_sentences_per_paragraph >= 1, (
            "`--min_sentences_per_paragraph` must be a positive integer"
        )
        assert self.hparams.min_paragraphs_per_document >= 1, (
            "`--min_paragraphs_per_document` must be a positive integer"
        )

        assert self.hparams.paragraph_ratio is None or (0.0 <= self.hparams.paragraph_ratio <= 1.0), (
            "`--paragraph_ratio` must be a float in [0.0, 1.0]"
        )
        assert 0.0 <= self.hparams.document_ratio <= 1.0, (
            "`--document_ratio` must be a float in [0.0, 1.0]"
        )

        self.actual = None

    @abstractmethod
    def make(self, documents: Generator[List[List[str]], None, None]) -> Generator[Dict, None, None]:
        r""" Create and yield examples that will be returned. """

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        r""" Process a batch of texts. """
        documents = (b[self.hparams.field] for b in batch)
        documents = clean_documents(
            documents=documents,
            paragraph_separator=self.hparams.paragraph_separator,
            min_sentence_length=self.hparams.min_sentence_length,
            min_paragraph_length=self.hparams.min_paragraph_length,
            min_document_length=self.hparams.min_document_length,
            min_sentences_per_paragraph=self.hparams.min_sentences_per_paragraph,
            min_paragraphs_per_document=self.hparams.min_paragraphs_per_document,
        )
        examples = list(self.make(documents))
        return examples

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        super(_ParagraphStrategy, _ParagraphStrategy).add_arguments_to_argparse(parser)
        parser.add_argument('--paragraph_separator', default='\n\n', required=False,
                            help="Split documents into paragraphs on this characted (string)")
        parser.add_argument('--min_sentence_length', type=int, default=20, required=False,
                            help="Minimum length to consider a sentence (in characters)")
        parser.add_argument('--min_paragraph_length', type=int, default=60, required=False,
                            help="Minimum length to consider a paragraph (in characters)")
        parser.add_argument('--min_document_length', type=int, default=200, required=False,
                            help="Minimum length to consider a document (in characters)")
        parser.add_argument('--min_sentences_per_paragraph', type=int, default=1, required=False,
                            help="Minimum number of cleaned sentences per paragraph (in characters)")
        parser.add_argument('--min_paragraphs_per_document', type=int, default=1, required=False,
                            help="Minimum number of cleaned paragraphs per document (in characters)")
        parser.add_argument(
            '--paragraph_ratio',
            type=float,
            default=None,
            help=(
                "How many paragraphs per documents should be used as pivot. "
                "None means 1 per document. A float between 0.0 and 1.0 "
                "means the corrisponding percentage of documents. "
                "With 0.0, no output pairs will be created."
            )
        )
        parser.add_argument(
            '--document_ratio',
            type=float,
            default=1.0,
            help=(
                "How many documents should be considered. None means all. "
                "A float will be used as probability to select a document. "
                "Negatives may anyway considered discarded documents."
            )
        )


class _PairwiseStrategy(_ParagraphStrategy):

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

        if len(self.hparams.max_negatives) == 1:
            self.hparams.max_negatives = (self.hparams.max_negatives[0], self.hparams.max_negatives[0])

        if self.hparams.max_hard_negatives is not None and len(self.hparams.max_hard_negatives) == 1:
            self.hparams.max_hard_negatives = (self.hparams.max_hard_negatives[0], self.hparams.max_hard_negatives[0])

        assert len(self.hparams.max_negatives) == 2, (
            "`--max_negatives` must be an integer or a range"
        )
        assert self.hparams.max_hard_negatives is None or len(self.hparams.max_hard_negatives) == 2, (
            "`--max_hard_negatives` must be an integer or a range"
        )

    def get_random_max_negatives(self) -> int:
        r""" Get random number of negatives. """
        return random.randint(*self.hparams.max_negatives)

    def get_random_max_hard_negatives(self) -> int:
        r""" Get random number of hard negatives. """
        if self.hparams.max_hard_negatives is None:
            raise ValueError("Cannot call `get_random_max_hard_negatives` without setting `max_hard_negatives`")
        return random.randint(*self.hparams.max_hard_negatives)

    def extract_random_span(self, paragraph: List[str], length: int, return_remain: bool = False):
        r""" Extract a random span of sentences from a paragraph. """
        position = random.randint(0, len(paragraph) - length)
        if return_remain:
            res = paragraph[:position] + paragraph[position + length:]
            return paragraph[position:position + length], res
        else:
            return paragraph[position:position + length]

    def extract_two_random_spans(
        self, paragraph: List[str], length_1: int, length_2: int, return_remain: bool = False
    ) -> Tuple:
        r""" Extract a random span of sentences from a paragraph. """
        position_1 = random.randint(0, len(paragraph) - (length_1 + length_2))
        position_2 = random.randint(position_1 + length_1, len(paragraph) - length_2)
        if return_remain:
            res = (
                paragraph[:position_1] \
                + paragraph[position_1 + length_1:position_2] \
                + paragraph[position_2 + length_2:]
            )
            return paragraph[position_1:position_1 + length_1], paragraph[position_2:position_2 + length_2], res
        else:
            return paragraph[position_1:position_1 + length_1], paragraph[position_2:position_2 + length_2]

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        super(_PairwiseStrategy, _PairwiseStrategy).add_arguments_to_argparse(parser)
        parser.add_argument('--max_negatives', default=None, required=True, type=int, nargs='+',
                            help="Max number or range of all negatives.")
        parser.add_argument('--max_hard_negatives', default=None, required=False, type=int, nargs='+',
                            help="Max number or range of hard negatives (from same document).")


class _SortingStrategy(_ParagraphStrategy):

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

        assert 0 < self.hparams.number_range[0] <= self.hparams.number_range[1], (
            "`--number_range` must be a non-empty range with limits greater than 0"
        )

        self.hparams.number_range = tuple(range(self.hparams.number_range[0], self.hparams.number_range[1] + 1))

        if self.hparams.number_probs is None:
            self.hparams.number_probs = (1.0, ) * len(self.hparams.number_range)

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        super(_SortingStrategy, _SortingStrategy).add_arguments_to_argparse(parser)
        parser.add_argument('--number_range', required=True, type=int, nargs=2,
                            help="Min and max number of paragraphs to order.")
        parser.add_argument('--number_probs', default=None, required=False, type=float, nargs='+',
                            help="Probabilities of paragraphs numbers.")
