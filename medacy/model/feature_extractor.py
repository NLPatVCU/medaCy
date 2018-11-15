from spacy.tokens.underscore import Underscore

class FeatureExtractor:
    """
    Extracting training data for use in a CRF.
    Features are given as rich dictionaries as described in:
    https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#features

    sklearn CRF suite is a wrapper for CRF suite that gives it a sci-kit compatability.
    """

    def __init__(self, window_size=2, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num']):
        """

        :param window_size: window size to pull features from on a given token, default 2 on both sides.
        :param spacy_features: Default token attributes that spacy sets to utilize as features
        """
        self.window_size = window_size
        self.all_custom_features = list(Underscore.token_extensions.keys()) #do not ask how long this took to find.
        self.spacy_features = spacy_features

    def __call__(self, doc):

        features = [self._sent_to_feature_dicts(sent) for sent in doc.sents]
        labels = [self._sent_to_labels(sent) for sent in doc.sents]

        return (features, labels)

    def get_features_with_span_indices(self, doc):

        features = [self._sent_to_feature_dicts(sent) for sent in doc.sents]

        indices = [[(token.idx, token.idx+len(token)) for token in sent] for sent in doc.sents]

        return (features, indices)



    def _sent_to_feature_dicts(self, sent):
        return [self._token_to_feature_dict(i, sent) for i in range(len(sent))]

    def _sent_to_labels(self, sent, attribute='gold_label'):
        return [token._.get(attribute) for token in sent]

    def mapper_for_crf_wrapper(self, text):
        """
        CRF wrapper uses regexes to extract the output of the underlying C++ code.
        The inclusion of \n and space characters mess up these regexes, hence we map them to text here.
        :return:
        """
        if text == r"\n":
            return "#NEWLINE"
        if text == r"\t":
            return "#TAB"
        if text == " ":
            return "#SPACE"
        if text == "":
            return "#EMPTY"

        return text


    def _token_to_feature_dict(self, index, sentence):
        """

        :param index: the index of the token in the sequence
        :param sentence: an array of tokens corresponding to a sequence
        :return:
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
                current.update({'%i:%s' % (i, feature) : getattr(token,feature) for feature in self.spacy_features})



                features.update(current)

        return features





