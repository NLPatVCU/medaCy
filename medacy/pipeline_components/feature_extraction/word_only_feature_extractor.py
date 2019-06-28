"""
One Hot Feature extraction for testing new networks.
"""
from spacy.tokens import Token
from itertools import cycle

import sys

class WordOnlyFeatureExtractor:
    # TODO Remove segmentation if it ends up unused
    segment_size = 100

    def __init__(self, segment_size):
        self.segment_size = segment_size

    def __call__(self, doc, file_name):
        """
        Extract features, labels, and corresponding spans from a document

        :param doc: Annotated Spacy Doc object
        :return: List of tuples of form:
            [(feature dictionaries for sequence, indices of tokens in seq, document label)]
        """
        features = [self.sequence_to_features(sent) for sent in doc.sents]
        labels = [self.sequence_to_labels(sent) for sent in doc.sents]
        indices = [[(token.idx, token.idx+len(token)) for token in sent] for sent in doc.sents]

        # Need to fix model so we don't have to return redundant data
        # features = list(zip(features, [None for _ in range(len(features))]))
        features = list(zip(features, indices, cycle([file_name])))

        return features, labels

    def sequence_to_features(self, sequence):
        """
        Transforms a given sequence of spaCy token objects into a discrete feature dictionary for us in a CRF.

        :param sequence:
        :return: a sequence of feature dictionaries corresponding to the token.
        """
        return [str(token) for token in sequence]

    def sequence_to_labels(self, sequence, attribute='gold_label'):
        """

        :param sequence: a sequence of tokens to retrieve labels from
        :param attribute: the name of the attribute that is holding the tokens label. This defaults to 'gold_label' which was set in the GoldAnnotator Component.
        :return: a list of token labels.
        """
        return [token._.get(attribute) for token in sequence]

    def get_segments(self, source):
        segment_size = self.segment_size
        segments = []

        for i in range(0, len(source), segment_size):
            segments.append(source[i:i + segment_size])

        return segments

    def get_features_with_span_indices(self, doc):
        """
        Given a document this method orchestrates the organization of features and labels for the sequences to classify.
        Sequences for classification are determined by the sentence boundaries set by spaCy. These can be modified.

        :param doc: an annoted spacy Doc object
        :return: Tuple of parallel arrays - 'features' an array of feature dictionaries for each sequence (spaCy determined sentence)
        and 'indices' which are arrays of character offsets corresponding to each extracted sequence of features.
        """
        sequences = self.get_segments(doc)

        features = [self.sequence_to_features(sequence) for sequence in sequences]

        indices = [[(token.idx, token.idx+len(token)) for token in sequence] for sequence in sequences]

        return features, indices
