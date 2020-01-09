"""
Extracting training data for use in the BERT Learner
"""
from itertools import cycle

class TextExtractor:
    """Text Extractor. Only extracts the text itself so that BERT can handle the rest. Usable
    with any other class that only requires the token text for features.
    """

    def __call__(self, doc, file_name):
        """
        Extract token text from document.

        :param doc: Annotated Spacy Doc object
        :param file_name: Filename to associate these sequences with
        :return: List of tuples of form:
            [(feature dictionaries for sequence, indices of tokens in seq, document label)]
        """
        features = [[token.text for token in sent] for sent in doc.sents]
        labels = [[token._.get('gold_label') for token in sent]for sent in doc.sents]
        indices = [[(token.idx, token.idx+len(token)) for token in sent] for sent in doc.sents]

        features = list(zip(features, indices, cycle([file_name])))
        return features, labels
