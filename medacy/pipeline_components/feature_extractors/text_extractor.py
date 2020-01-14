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

    def get_features_with_span_indices(self, doc):
        """
        Given a document this method orchestrates the organization of features and labels for the sequences to classify.
        Sequences for classification are determined by the sentence boundaries set by spaCy. These can be modified.

        :param doc: an annotated spaCy Doc object
        :return: Tuple of parallel lists, a list of token texts and a list of corresponding character spans
        """

        features = [token.text for token in doc]
        indices = [(token.idx, token.idx + len(token)) for token in doc]
        return features, indices
