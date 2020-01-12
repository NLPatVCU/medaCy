"""
Extracting training data for use in the BERT Learner
"""
from itertools import cycle

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor


class TextExtractor(FeatureExtractor):
    """
    Text Extractor. Only extracts the text itself so that BERT can handle the rest. Usable
    with any other class that only requires the token text for features.
    """

    def __call__(self, doc):
        """
        Extract token text from document.

        :param doc: Annotated spaCy Doc object
        :return: List of tuples of form:
            [(feature dictionaries for sequence, indices of tokens in seq, document label)]
        """
        features = [[token.text for token in sent] for sent in doc.sents]
        labels = [[token._.get('gold_label') for token in sent]for sent in doc.sents]
        indices = [[(token.idx, token.idx+len(token)) for token in sent] for sent in doc.sents]

        file_name = doc._.file_name

        features = list(zip(features, indices, cycle([file_name])))
        return features, labels

    def _token_to_feature_dict(self, index, sentence):
        """
        :param index: The position of the target token in the sentence
        :param sentence: An iterable, subscriptable container of Tokens
        :return: a list of token texts in the window size
        """

        token_texts = []

        for i in range(-self.window_size, self.window_size + 1):
            # loop through our window, ignoring tokens that aren't there
            if not 0 <= index + i < len(sentence):
                continue
            token = sentence[index + i]
            token_texts.append(token.text)
