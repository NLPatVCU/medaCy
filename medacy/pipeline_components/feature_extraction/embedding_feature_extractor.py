from spacy.tokens import Token
from .discrete_feature_extractor import FeatureExtractor


class EmbeddingFeatureExtractor(FeatureExtractor):
    """A feature extractor that overlays the distance between word vectors."""

    def __init__(self, vectors, window_size=2, spacy_features=None):
        super().__init__(
            window_size=window_size,
            spacy_features=spacy_features
        )
        self.vectors = vectors

    def _token_to_feature_dict(self, index, sentence):
        """
        Extracts features of a given token.

        This differs from the function in the super class in that it adds as a feature the distance between the vector
        of the word at the given index and all other words in the window.

        :param index: the index of the token in the sequence
        :param sentence: an array of tokens corresponding to a sequence
        :return: a dictionary with a feature representation of the spaCy token object.
        """

        # This should automatically gather features that are set on tokens
        # by looping over all attributes set on sentence[index] that begin with 'feature'

        named_entity = sentence[index].text
        if named_entity not in self.vectors.keys():
            return super()._token_to_feature_dict(index, sentence)

        features = {
            'bias': 1.0
        }
        for i in range(-self.window_size, self.window_size+1):  # loop through our window
            if 0 <= (index + i) < len(sentence):  # for each index in the window size
                token = sentence[index+i]

                # adds features from medaCy pipeline
                current = {'%i:%s' % (i, feature) : token._.get(feature) for feature in self.all_custom_features}

                # adds features that are overlayed from spaCy token attributes
                for feature in self.spacy_features:
                    if isinstance(getattr(token, feature), Token):
                        current.update({'%i:%s' % (i, feature) : getattr(token, feature).text})
                    else:
                        current.update({'%i:%s' % (i, feature) : getattr(token, feature)})

                similarity = self.vectors.similarity(named_entity, token)

                current.update({
                    f"{i}:similarity": similarity
                })

                # Extract features from the vector representation of this token
                features.update(current)

        return features
