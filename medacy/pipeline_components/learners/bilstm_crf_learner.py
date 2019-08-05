"""
BiLSTM+CRF PyTorch network and model.
"""
import logging
import random
import string
import unicodedata
import re

from gensim.models import KeyedVectors
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from medacy.nn import BiLstmCrf

class BiLstmCrfLearner:
    """
    BiLSTM-CRF model class for using the network. Currently handles all vectorization as well.

    :ivar device: Pytorch device to use.
    :ivar model: Instance of BiLstmCrfNetwork to use.
    :ivar tag_to_index: Tag to index dictionary for vectorization.
    :ivar untrained_tokens: Out of vocabulary tokens word embeddings analysis during debugging.
    :ivar other_features: Names of other word features being used.
    :ivar window_size: Range of surrounding word's features that were extracted.
    :ivar word_embeddings_file: File to load word embeddings from.
    :ivar word_vectors: Gensim word vectors object for use in configuring word embeddings.
    """

    def __init__(self, word_embeddings, cuda_device):
        """Init BiLstmCrfLearner object.

        :param word_embeddings: Path to word embeddings file to use.
        :param cuda_device: Index of cuda device to use. Use -1 to use CPU.
        """
        if word_embeddings is None:
            raise ValueError('BiLSTM-CRF requires word embeddings.')

        torch.manual_seed(1)
        self.character_to_index = {
            character:(index + 1) for index, character in enumerate(string.printable)
        }
        self.word_embeddings_file = word_embeddings

        device_string = 'cuda:%d' % cuda_device if cuda_device >= 0 else 'cpu'
        self.device = torch.device(device_string)

        # Other instance attributes
        self.word_vectors = None
        self.untrained_tokens = set()
        self.other_features = []
        self.tag_to_index = {}
        self.window_size = 0
        self.model = None

        # TODO: Implement cleaner way to handle this
        self.learning_rate = 0.01
        if word_embeddings.endswith('test_word_embeddings.txt'):
            self.epochs = 2
            self.crf_delay = 1
        else:
            self.epochs = 40
            self.crf_delay = 20

    def load_word_embeddings(self):
        """Uses self.word_embeddings_file and gensim to load word embeddings into memory.
        """
        embeddings_file = self.word_embeddings_file
        is_binary = True if self.word_embeddings_file[-4:] == '.bin' else False
        self.word_vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=is_binary)

    def find_other_features(self, example):
        """Get the names of the other word features being used.

        :param example: One set of features to search through for the names.
        """
        if '0:text' not in example:
            raise ValueError('BiLSTM-CRF requires the "0:text" spaCy feature.')

        # Find other feature names
        for key in example:
            if key[:2] == '0:' and key != '0:text':
                self.other_features.append(key[2:])

    def devectorize_tag(self, tag_indices):
        """Devectorize a list of tag indices using self.tag_to_index

        :param tag_indices: List of tag indices.
        :return: List of tags.
        """
        to_tag = {y:x for x, y in self.tag_to_index.items()}
        tags = [to_tag[index] for index in tag_indices[0]]
        return tags

    def create_tag_dictionary(self, tags):
        """Setup self.tag_to_index

        :param tags: List of list of tag names.
        """
        tag_to_index = {}

        for sequence in tags:
            for tag in sequence:
                if tag not in tag_to_index:
                    tag_to_index[tag] = len(tag_to_index)

        self.tag_to_index = tag_to_index

    def vectorize_tags(self, tags):
        """Convert list of tag names into their indices.

        :param tags: List of tags to convert.
        :returns: Torch tensor of indices.
        """
        indices = []

        for tag in tags:
            if tag in self.tag_to_index:
                indices.append(self.tag_to_index[tag])

        return torch.tensor(indices, dtype=torch.long, device=self.device)

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

    def find_window_indices(self, token):
        """Get relative indexes of window words. Avoids trying to access keys that don't exist.

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
            for i in window_range:
                if i in window:
                    for feature in self.other_features:
                        key = '%d:%s' % (i, feature)
                        feature = float(token[key])
                        token_vector.append(feature)
                else:
                    for feature in self.other_features:
                        feature = 0.0
                        token_vector.append(feature)

            tokens_vector.append(token_vector)

        return tokens_vector

    def fit(self, x_data, y_data):
        """Fully train model based on x and y data. self.model is set to trained model.

        :param x_data: List of list of tokens.
        :param y_data: List of list of correct labels for the tokens.
        """
        if self.word_vectors is None:
            self.load_word_embeddings()

        self.create_tag_dictionary(y_data)

        # Find other feature names
        self.find_other_features(x_data[0][0])

        # Calculate window size
        self.find_window_size(x_data)

        # Vectorize data
        sentences = []
        correct_tags = []
        for sentence, sentence_tags in zip(x_data, y_data):
            tokens_vector = self.vectorize_tokens(sentence)
            correct_tags_vector = self.vectorize_tags(sentence_tags)
            sentences.append(tokens_vector)
            correct_tags.append(correct_tags_vector)
        data = list(zip(sentences, correct_tags))

        # Create network
        model = BiLstmCrf(
            self.word_vectors,
            len(self.other_features) * (self.window_size * 2 + 1),
            self.tag_to_index,
            self.device
        )

        # Move to GPU if possible
        if self.device.type != 'cpu':
            logging.info('CUDA available. Moving model to GPU.')
            model = model.to(self.device)

        # Setup optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_function = nn.NLLLoss()

        logging.info('Training BiLSTM...')

        # Training loop
        for i in range(1, self.epochs + 1):
            random.shuffle(data)
            epoch_losses = []

            for sentence, sentence_tags in data:
                if i <= self.crf_delay:
                    emissions = model(sentence)
                    predictions = F.log_softmax(emissions, dim=1)
                    loss = loss_function(predictions, sentence_tags)
                else:
                    emissions = model(sentence).unsqueeze(1)
                    sentence_tags = sentence_tags.unsqueeze(1)
                    loss = -model.crf(emissions, sentence_tags)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)

            average_loss = sum(epoch_losses) / len(epoch_losses)
            logging.info('Epoch %d average loss: %f', i, average_loss)
            logging.debug(self.untrained_tokens)

        self.model = model

    def predict(self, sequences):
        """Use model to make predictions over a given dataset.

        :param sequences: Sequences to predict labels for.
        :return: List of list of predicted labels.
        """
        if not self.word_vectors:
            raise RuntimeError('Loading word embeddings is required.')

        with torch.no_grad():
            predictions = []
            for sequence in sequences:
                vectorized_tokens = self.vectorize_tokens(sequence)
                emissions = self.model(vectorized_tokens).unsqueeze(1)
                tag_indices = self.model.crf.decode(emissions)
                predictions.append(self.devectorize_tag(tag_indices))

        return predictions

    def save(self, path):
        """Save model and other required values.

        :param path: Path to save model to.
        """
        properties = {
            'model': self.model,
            'tag_to_index': self.tag_to_index,
            'other_features': self.other_features,
            'window_size': self.window_size
        }

        if path[-4:] != '.pth':
            path += '.pth'

        torch.save(properties, path)

    def load(self, path):
        """Load model and other required values from given path.

        :param path: Path of saved model.
        """
        saved_data = torch.load(path)

        self.tag_to_index = saved_data['tag_to_index']
        self.other_features = saved_data['other_features']
        self.window_size = saved_data['window_size']

        model = saved_data['model']
        model.eval()
        self.model = model
        self.load_word_embeddings()
