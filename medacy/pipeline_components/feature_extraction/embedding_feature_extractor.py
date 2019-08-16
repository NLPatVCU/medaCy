from spacy.tokens import Token
from .discrete_feature_extractor import FeatureExtractor


class EmbeddingFeatureExtractor(FeatureExtractor):
    """A feature extractor that overlays the distance between word vectors."""

    def __init__(self, vectors, window_size=2, spacy_features=None, use_embedding=True, use_distance=True):
        """
        Note that custom medaCy features are pulled from spaCy custom token attributes that begin with 'feature_'.

        :param vectors: an already-loaded gensim KeyedVectors object
        :param window_size: window size to pull features from on a given token, default 2 on both sides.
        :param spacy_features: Default token attributes that spaCy sets to utilize as features (list)
        """
        super().__init__(
            window_size=window_size,
            spacy_features=spacy_features
        )
        self.vectors = vectors
        self.use_embedding = use_embedding
        self.use_distance = use_distance

    def _token_to_feature_dict(self, index, sentence):
        """
        Extracts features of a given token.

        This differs from the function in the super class in that it adds as a feature the distance between the vector
        of the word at the given index and all other words in the window.

        :param index: the index of the token in the sequence
        :param sentence: an array of tokens corresponding to a sequence
        :return: a dictionary with a feature representation of the spaCy token object.
        """

        features = {
            'bias': 1.0
        }

        # Get the text of entity and see if it's in the vocabulary, else use standard feature extractor for this ent
        named_entity = sentence[index].text
        try:
            self.vectors[named_entity]  # We're just seeing if it's there
        except KeyError:
            return super()._token_to_feature_dict(index, sentence)

        # This should automatically gather features that are set on tokens
        # by looping over all attributes set on sentence[index] that begin with 'feature'

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

                if self.use_distance:
                    # Try and get the similarity to each word in the window, else set similarity to zero
                    try:
                        similarity = self.vectors.similarity(named_entity, token.text)
                    except KeyError:
                        similarity = 0

                    current.update({
                        f"{i}:similarity": similarity
                    })

                if self.use_embedding:
                    # Try and get the word vector, then make every index its own feature
                    try:
                        word_embedding = self.vectors[token.text]
                        word_features = {}
                        for n, idx in enumerate(float(x) for x in word_embedding):
                            word_features[f"{i}:embedding-{idx}"] = n
                        current.update(word_features)
                    except KeyError:
                        pass

                # Extract features from the vector representation of this token
                features.update(current)

        return features
