"""
Vectorizer for medaCy PyTorch classes.
"""
import re
import string

import torch
import unicodedata
from gensim.models import KeyedVectors


class Vectorizer:
    """Vectorizer for medaCy PyTorch data. Contains encoding methods and tracking encoding values.

    :ivar device: PyTorch device to use.
    :ivar word_vectors: Gensim Word2VecKeyedVectors for word embeddings.
    :ivar untrained_tokens: Out of vocabulary tokens.
    :ivar other_features: Features other than word embeddings or ids.
    :ivar window_size: Number of tokens to include on either side of current token.
    :ivar tag_to_index: Dictionary of label to id mappings.
    :ivar character_to_index: Dictionary of character to id mappings.
    """
    def __init__(self, device):
        """Initialize Vectorizer.

        :param device: PyTorch device to use.
        """
        self.device = device
        self.word_vectors = None
        self.untrained_tokens = set()
        self.other_features = {}
        self.window_size = 0
        self.tag_to_index = {}

        self.character_to_index = {
            character: index for index, character in enumerate(string.printable, 1)
        }

    def load_word_embeddings(self, embeddings_file):
        """Uses self.word_embeddings_file and gensim to load word embeddings into memory.

        :param embeddings_file: Word embeddings file to use. Can be .bin or other common formats.
        """
        is_binary = embeddings_file.endswith('.bin')
        word_vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=is_binary)
        self.word_vectors = word_vectors

    def create_tag_dictionary(self, tags):
        """Setup self.tag_to_index

        :param tags: List of list of tag names. Usually all true labels for a dataset.
        """
        tag_to_index = {}

        for sequence in tags:
            for tag in sequence:
                if tag not in tag_to_index:
                    tag_to_index[tag] = len(tag_to_index)

        self.tag_to_index = tag_to_index

    def add_tag(self, tag):
        """Add tag to self.tag_to_index

        :param tag: Tag to add.
        """
        self.tag_to_index[tag] = len(self.tag_to_index)

    def create_feature_dictionary(self, feature_name, sentences):
        """Get dictionary that maps all possible values of a specific feature to ids.

        :param feature_name: Name of feature.
        :param sentences: Sentences to get feature for.
        :return: Dictionary for given feature.
        """
        feature_to_index = {}
        feature_name = '0:' + feature_name

        for sentence in sentences:
            for token in sentence:
                feature = token[feature_name]
                if feature not in feature_to_index:
                    feature_to_index[feature] = len(feature_to_index)

        return feature_to_index

    def find_other_features(self, example):
        """Get the names of the other word features being used.

        :param example: One set of features to search through for the names.
        """
        if '0:text' not in example:
            raise ValueError('BiLSTM-CRF requires the "0:text" spaCy feature.')

        # Find other feature names
        for key in example:
            if key.startswith('0:') and key != '0:text':
                feature = key[2:]
                self.other_features[feature] = {}

    def find_window_size(self, x_data):
        """Find and set the window size based on input data. Only supports single digit window
        sizes.

        :param x_data: Input data to use.
        """
        # Find longest sequence and use token in center for analysis
        test_token = None
        longest_length = 0
        for sentence in x_data:
            if len(sentence) > longest_length:
                longest_length = len(sentence)
                test_token = sentence[int(longest_length/2)]

        lowest = 0
        highest = 0

        # Loop through keys in test token to find highest and lowest window distances.
        for key in test_token:
            if key[0] == '-':
                index = int(key[:2])
                if index < lowest:
                    lowest = index
            elif key[0].isnumeric():
                index = int(key[0])
                if index > highest:
                    highest = index

        assert -lowest == highest, 'Word feature window is asymmetrical'

        self.window_size = highest

    def unicode_to_ascii(self, unicode_string):
        """Convert unicode string to closest ASCII equivalent. Based on code found at:
        https://stackoverflow.com/a/518232/2809427

        :param unicode_string: String to convert to ASCII
        :return: String with every character converted to most similar ASCII character.
        """
        unicode_string = re.sub(u"\u2013", "-", unicode_string) # em dash

        return ''.join(
            character for character in unicodedata.normalize('NFD', unicode_string)
            if unicodedata.category(character) != 'Mn'
            and character in string.printable
        )

    def devectorize_tag(self, tag_indices):
        """Devectorize a list of tag indices using self.tag_to_index

        :param tag_indices: List of tag indices.
        :return: List of tags.
        """
        to_tag = {y:x for x, y in self.tag_to_index.items()}
        tags = [to_tag[index] for index in tag_indices]
        return tags

    def find_window_indices(self, token):
        """Get relative indices of window words. Avoids trying to access keys that don't exist.

        :param token: Token the indexes are relative to.
        :return: List of indices
        """
        window = []
        window_range = range(-self.window_size, self.window_size + 1)

        for i in window_range:
            test_key = 'text'
            test_key = '%d:%s' % (i, test_key)
            if test_key in token:
                window.append(i)

        return window

    def one_hot(self, index_dictionary, value):
        """
        Create a one-hot vector representation for discrete features that appear in the X_data
        :param index_dictionary: A dictionary mapping discrete features to unique integers (ie the order
            they appeared in the X_data; see self.create_feature_dictionary)
        :param value: The discrete feature
        :return: A one-hot vector for that discrete feature
        """
        vector = [0.0] * len(index_dictionary)

        if value in index_dictionary:
            index = index_dictionary[value]
            vector[index] = 1.0

        return vector

    def vectorize_tokens(self, tokens):
        """Vectorize list of tokens.

        :param tokens: Tokens to vectorize.
        :return: List of vectors.
        """
        tokens_vector = []

        for token in tokens:
            token_vector = []

            # Add text index for looking up word embedding
            token_text = token['0:text']
            token_text = self.unicode_to_ascii(token_text)

            # Look up word embedding index
            try:
                embedding_index = self.word_vectors.vocab[token_text].index
            except KeyError:
                embedding_index = len(self.word_vectors.vocab)

                # Only for logging untrained tokens
                self.untrained_tokens.add(token_text)

            token_vector.append(embedding_index)

            # Add list of character indices as second item
            character_indices = []
            for character in token_text:
                index = self.character_to_index[character]
                character_indices.append(index)

            # If there were no indices ex. special characters only
            if not character_indices:
                # Append the padding index
                character_indices.append(0)
            token_vector.append(character_indices)

            # Find window indices
            window = self.find_window_indices(token)

            # Add features to vector in order
            window_range = range(-self.window_size, self.window_size + 1)
            other_feature_names = [key for key in self.other_features]
            other_feature_names.sort()

            for i in window_range:
                if i in window:
                    for feature_name in other_feature_names:
                        key = '%d:%s' % (i, feature_name)
                        feature = token[key]
                        vector = self.one_hot(self.other_features[feature_name], feature)
                        token_vector.extend(vector)
                else:
                    for feature in self.other_features:
                        vector = [0.0] * len(self.other_features[feature])
                        token_vector.extend(vector)

            tokens_vector.append(token_vector)

        return tokens_vector

    def vectorize_tags(self, tags):
        """Convert list of tag names into their indices.

        :param tags: List of tags to convert.
        :return: Torch tensor of indices.
        """
        indices = [self.tag_to_index[tag] for tag in tags]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def vectorize_dataset(self, x_data, y_data):
        """Vectorize entire dataset.

        :param x_data: Sequences.
        :param y_data: True labels.
        :return: Vectorized data.
        """
        self.create_tag_dictionary(y_data)

        # Find other feature names
        self.find_other_features(x_data[0][0])

        # Calculate window size
        self.find_window_size(x_data)

        # Create feature dictionaries
        for feature in self.other_features:
            self.other_features[feature] = self.create_feature_dictionary(feature, x_data)

        # Vectorize data
        sentences = []
        correct_tags = []
        for sentence, sentence_tags in zip(x_data, y_data):
            tokens_vector = self.vectorize_tokens(sentence)
            correct_tags_vector = self.vectorize_tags(sentence_tags)
            sentences.append(tokens_vector)
            correct_tags.append(correct_tags_vector)
        data = list(zip(sentences, correct_tags))

        return data

    def get_values(self):
        """Get Vectorizer values so they can saved or migrated.

        :return: Values to get.
        """
        values = {
            'tag_to_index': self.tag_to_index,
            'character_to_index': self.character_to_index,
            'untrained_tokens': self.untrained_tokens,
            'window_size': self.window_size,
            'other_features': self.other_features
        }

        return values

    def load_values(self, values):
        """Load saved Vectorizer values into this object.
        :param values: Values to load.
        """
        self.tag_to_index = values['tag_to_index']
        self.untrained_tokens = values['untrained_tokens']
        self.character_to_index = values['character_to_index']
        self.window_size = values['window_size']
        try:
            self.other_features = values['other_features']
        except KeyError:
            raise Exception('Tried to load deprecated Medacy model')
