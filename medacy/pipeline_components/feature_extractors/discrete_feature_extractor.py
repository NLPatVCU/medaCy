from spacy.tokens import Token
from spacy.tokens.underscore import Underscore

from medacy.pipeline_components.feature_extractors import FeatureTuple


class FeatureExtractor:
    """
    This class allows for full control of both spaCy features that exist on tokens
    and custom medaCy overlayed features. The current implementation is designed solely for use with sequence
    classifiers such as discriminative conditional random fields.

    Custom medaCy features are pulled from spaCy custom token attributes that begin with 'feature_'.
    """

    def __init__(self, window_size=2, spacy_features=None):
        """
        :param window_size: window size to pull features from on a given token, default 2 on both sides.
        :param spacy_features: Default token attributes that spaCy sets to utilize as features
        """
        self.window_size = window_size
        self.all_custom_features = [attr for attr in list(Underscore.token_extensions.keys()) if attr.startswith('feature_')]
        self.spacy_features = spacy_features or ['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num']

    def __call__(self, doc):
        """
        Extract features, labels, and corresponding spans from a document

        :param doc: Annotated spaCy Doc object
        :return: List of tuples of form: [(feature dictionaries for sequence, indices of tokens in seq, document label)]
        """

        features = [self._sequence_to_feature_dicts(sent) for sent in doc.sents]
        labels = [self._sequence_to_labels(sent) for sent in doc.sents]
        indices = [[(token.idx, token.idx + len(token)) for token in sent] for sent in doc.sents]

        file_name = doc._.file_name
        features = [FeatureTuple(*t, file_name) for t in zip(features, indices)]

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
        indices = [[(token.idx, token.idx + len(token)) for token in sent] for sent in doc.sents]
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

        # This should automatically gather features that are set on tokens
        # by looping over all attributes set on sentence[index] that begin with 'feature'

        features = {
            'bias': 1.0
        }

        for i in range(-self.window_size, self.window_size+1):
            # loop through our window, ignoring tokens that aren't there
            if not 0 <= index + i < len(sentence):
                continue

            token = sentence[index+i]

            # adds features from medacy pipeline
            current = {f'{i}:{feature}': token._.get(feature) for feature in self.all_custom_features}

            # adds features that are overlayed from spacy token attributes
            for feature in self.spacy_features:
                if isinstance(getattr(token, feature), Token):
                    current[f'{i}:{feature}'] = getattr(token, feature).text
                else:
                    current[f'{i}:{feature}'] = getattr(token, feature)

            features.update(current)

        return features
