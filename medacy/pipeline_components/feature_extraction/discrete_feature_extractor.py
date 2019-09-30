"""
Extracting training data for use in a CRF.
Features are extracted as discrete dictionaries as described in
`sklearn-crfsuite <https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#features>`_.


These extracted features CANNOT be used in sequence to sequence models expecting continuous inputs (e.g. word vectors).

`sklearn-crfsuite <https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#features>`_ is a wrapper for a C CRF implementation that gives it a sci-kit compatability.
"""
from itertools import cycle

from spacy.tokens import Token
from spacy.tokens.underscore import Underscore


class FeatureExtractor:

    def __init__(self, window_size=2, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num']):
        """
        Initializes a FeatureExtractor. This class allows for full control of both spacy features that exist on tokens
        and custom medacy overlayed features. The current implementation is designed solely for use with sequence
        classifiers such as discriminative conditional random fields.

        Custom medaCy features are pulled from spacy custom token attributes that begin with 'feature_'.

        :param window_size: window size to pull features from on a given token, default 2 on both sides.
        :param spacy_features: Default token attributes that spacy sets to utilize as features
        """
        self.window_size = window_size
        # do not ask how long this took to find.
        self.all_custom_features = [attribute for attribute in list(Underscore.token_extensions.keys()) if attribute.startswith('feature_')]
        self.spacy_features = spacy_features

    def __call__(self, doc, file_name):
        """
        Extract features, labels, and corresponding spans from a document

        :param doc: Annotated Spacy Doc object
        :param file_name: Filename to associate these sequences with
        :return: List of tuples of form:
            [(feature dictionaries for sequence, indices of tokens in seq, document label)]
        """

        features = [self._sequence_to_feature_dicts(sent) for sent in doc.sents]
        labels = [self._sequence_to_labels(sent) for sent in doc.sents]
        indices = [[(token.idx, token.idx+len(token)) for token in sent] for sent in doc.sents]

        features = list(zip(features, indices, cycle([file_name])))
        return features, labels

    def get_features_with_span_indices(self, doc):
        """
        Given a document this method orchestrates the organization of features and labels for the sequences to classify.
        Sequences for classification are determined by the sentence boundaries set by spaCy. These can be modified.

        :param doc: an annoted spacy Doc object
        :return: Tuple of parallel arrays - 'features' an array of feature dictionaries for each sequence (spaCy determined sentence)
        and 'indices' which are arrays of character offsets corresponding to each extracted sequence of features.
        """

        features = [self._sequence_to_feature_dicts(sent) for sent in doc.sents]

        indices = [[(token.idx, token.idx+len(token)) for token in sent] for sent in doc.sents]

        return features, indices



    def _sequence_to_feature_dicts(self, sequence):
        """
        Transforms a given sequence of spaCy token objects into a discrete feature dictionary for us in a CRF.

        :param sequence:
        :return: a sequence of feature dictionaries corresponding to the token.
        """
        return [self._token_to_feature_dict(i, sequence) for i in range(len(sequence))]

    def _sequence_to_labels(self, sequence, attribute='gold_label'):
        """

        :param sequence: a sequence of tokens to retrieve labels from
        :param attribute: the name of the attribute that is holding the tokens label. This defaults to 'gold_label' which was set in the GoldAnnotator Component.
        :return: a list of token labels.
        """
        return [token._.get(attribute) for token in sequence]

    def _token_to_feature_dict(self, index, sentence):
        """
        Extracts features of a given token

        :param index: the index of the token in the sequence
        :param sentence: an array of tokens corresponding to a sequence
        :return: a dictionary with a feature representation of the spaCy token object.
        """

        #This should automatically gather features that are set on tokens
        #by looping over all attributes set on sentence[index] that begin with 'feature'

        features = {
            'bias': 1.0
        }
        for i in range(-self.window_size, self.window_size+1): #loop through our window
            if 0 <= (index + i) and  (index + i) < len(sentence): #for each index in the window size
                token = sentence[index+i]

                #adds features from medacy pipeline
                current = {'%i:%s' % (i, feature) : token._.get(feature) for feature in self.all_custom_features}

                #adds features that are overlayed from spacy token attributes
                for feature in self.spacy_features:
                    if isinstance(getattr(token, feature), Token):
                        current.update({'%i:%s' % (i, feature) : getattr(token, feature).text})
                    else:
                        current.update({'%i:%s' % (i, feature) : getattr(token, feature)})

                # Extract features from the vector representation of this token

                features.update(current)

        return features
