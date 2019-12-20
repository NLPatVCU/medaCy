from spacy.tokens import Token

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor


class POSDropFeatureExtractor(FeatureExtractor):
    """
    This class functions similarly to FeatureExtractor, but when features are extracted for X data, words belonging
    to a given list of parts of speech are not included in the window.

    Given the sentence "The cow jumped over the moon", if a window size of two was used, prepositions and determiners
    were dropped, and the target word were 'moon', the words included in the window would be 'cow jumped moon'.
    """

    def __init__(self, window_size=2, spacy_features=None, ignored_pos=None):
        """
        :param window_size: window size to pull features from on a given token, default 2 on both sides.
        :param spacy_features: Default token attributes that spaCy sets to utilize as features
        :param ignored_pos: a list of parts of speech to ignore, using spaCy POS naming conventions
        """
        super().__init__(
            window_size=window_size,
            spacy_features=spacy_features
        )
        self.ignored_pos = ignored_pos or ['PREP']

    def _token_to_feature_dict(self, index, sentence):
        """
        Extracts features of a given token

        :param index: the index of the token in the sequence
        :param sentence: an array of tokens corresponding to a sequence
        :return: a dictionary with a feature representation of the spaCy token object.
        """

        # This should automatically gather features that are set on tokens
        # by looping over all attributes set on sentence[index] that begin with 'feature'

        target = sentence[index]
        sentence = [token for token in sentence if token.pos_ not in self.ignored_pos or token is target]

        features = {
            'bias': 1.0
        }

        for i in range(-self.window_size, self.window_size+1):
            # loop through our window, ignoring tokens that aren't there
            if not 0 <= index + i < len(sentence):
                continue

            token = sentence[index+i]

            # adds features from medacy pipeline
            current = {'%i:%s' % (i, feature): token._.get(feature) for feature in self.all_custom_features}

            # adds features that are overlayed from spacy token attributes
            for feature in self.spacy_features:
                if isinstance(getattr(token, feature), Token):
                    current['%i:%s' % (i, feature)] = getattr(token, feature).text
                else:
                    current['%i:%s' % (i, feature)] = getattr(token, feature)

            features.update(current)

        return features
