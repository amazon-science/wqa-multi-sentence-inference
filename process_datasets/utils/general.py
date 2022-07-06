import re
from typing import Any, Dict, Generator, Iterable, List

from blingfire import text_to_sentences


cleaner = re.compile(r"\s+")


def clean_sentences(sentences: List[str], min_sentence_length: int = 1) -> Generator[str, None, None]:
    r""" Check that sentences are long enough and non empty. """
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= min_sentence_length:
            yield sentence


def clean_paragraphs(
    paragraphs: List[str],
    min_paragraph_length: int = 1,
    min_sentences_per_paragraph: int = 1,
    min_sentence_length: int = 1,
) -> Generator[str, None, None]:
    r""" () is remainder after link in it was filtered out. """
    for paragraph in paragraphs:
        paragraphs = cleaner.sub(" ", paragraph.strip()).replace("()", "")
        if len(paragraph) >= min_paragraph_length:
            paragraph = list(
                clean_sentences(
                    text_to_sentences(paragraph).split("\n"), min_sentence_length=min_sentence_length
                )
            )
            if len(paragraph) >= min_sentences_per_paragraph:
                yield paragraph


def clean_documents(
    documents: Iterable[Dict],
    paragraph_separator: str = "\n\n",
    min_sentence_length: int = 1,
    min_sentences_per_paragraph: int = 1,
    min_paragraph_length: int = 1,
    min_paragraphs_per_document: int = 1,
    min_document_length: int = 1,
) -> List[List[List[str]]]:
    r""" Clean every document by splitting it in paragraphs and then by splitting each paragraph in sentences. """

    for document in documents:
        document = document.strip()
        if len(document) >= min_document_length:
            # generic filter on min length and special chars at the paragraph level
            document = list(
                clean_paragraphs(
                    re.split(paragraph_separator, document),
                    min_paragraph_length=min_paragraph_length,
                    min_sentence_length=min_sentence_length,
                    min_sentences_per_paragraph=min_sentences_per_paragraph,
                )
            )
            if len(document) >= min_paragraphs_per_document:
                yield document


def check_dict(dictionary: Dict):
    return all(v is not None for v in dictionary.values())


def cumsum_limit(values: Iterable[int], limit: int) -> int:
    r""" Return the position of the element which first violates the limit in the cumulative sum. """
    count = 0
    for i, v in enumerate(values):
        count += v
        if count > limit:
            return i
    return i


def dict2list(data: Dict[Any, List]) -> List[Dict]:
    r""" Convert a dict or lists to a list of dicts. """
    values = list(data.values())
    assert all(isinstance(v, list) for v in values)
    assert all(len(v) == len(values[0]) for v in values)

    if not data or all(len(v) == 0 for v in values):
        return []

    keys = data.keys()
    res = [
        {a: b for a, b in zip(keys, values)}
        for values in zip(*[data[key] for key in keys])
    ]
    return res


def list2dict(data: List[Dict]) -> Dict[Any, List]:
    r""" Convert a list of dicts to a dict of lists. """
    if not data:
        return {}

    assert all(isinstance(d, dict) for d in data)
    keys = data[0].keys()
    assert all(d.keys() == keys for d in data)

    res = {k: [d[k] for d in data] for k in keys}
    return res
