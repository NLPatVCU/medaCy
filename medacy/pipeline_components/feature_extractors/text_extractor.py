from medacy.pipeline_components.feature_extractors import FeatureTuple
from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from transformers import AdamW, BertTokenizer, BertForTokenClassification


class TextExtractor(FeatureExtractor):
    """
    Text Extractor. Only extracts the text itself so that BERT can handle the rest. Usable
    with any other class that only requires the token text for features.
    """

    def __init__(self):
        self.window_size = 0
        self.spacy_features = []

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
        features = [FeatureTuple(*t, file_name) for t in zip(features, indices)]

        return features, labels

    def get_features_with_span_indices(self, doc):
        """
        Given a document this method orchestrates the organization of features and labels for the sequences to classify.
        Sequences for classification are determined by the sentence boundaries set by spaCy. These can be modified.

        :param doc: an annotated spaCy Doc object
        :return: Tuple of parallel lists, a list of token texts and a list of corresponding character spans
        """

        """
        features = []
        for sent in doc.sents:
            features = [token.text for token in sent]
        """

        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

        features, indices = [], []
        for sent in doc.sents:
            feature = [token.text for token in sent]
            index = [(token.idx, token.idx + len(token)) for token in sent]
            encoded_sequence = [tokenizer.encode(token, add_special_tokens=False) for token in feature]
            encoded_sequence = tokenizer.build_inputs_with_special_tokens(encoded_sequence)
            bert_tokenized = []
            for x in encoded_sequence:
                if type(x) == int:
                    bert_tokenized.append(x)
                else:
                    bert_tokenized.extend(x)
            if len(bert_tokenized) > 200:
                num_splits = (len(bert_tokenized) // 500) + 3
                split_length = int(len(feature) / num_splits) + 1
                feature_splits = list(chunks(feature, split_length))
                indices_splits = list(chunks(index, split_length))
                features.extend(feature_splits)
                indices.extend(indices_splits)
            else:
                features.append(feature)
                indices.append(index)

        """        
        features = [[token.text for token in sent] for sent in doc.sents]
        indices = [[(token.idx, token.idx + len(token)) for token in sent] for sent in doc.sents]
        """
        return features, indices


def chunks(l, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
