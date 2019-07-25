"""
BiLSTM+CRF PyTorch network and model.

Original LSTM code was based off of tutorial found at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
Further customizations guided by:
- https://arxiv.org/pdf/1508.01991.pdf
- https://www.sciencedirect.com/science/article/pii/S1532046417300977
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
from torchcrf import CRF

# Constants
LEARNING_RATE = 0.01
HIDDEN_DIM = 200
CHARACTER_HIDDEN_DIM = 100
CHARACTER_EMBEDDING_SIZE = 100
EPOCHS = 40

class BiLstmCrfNetwork(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, wv, other_features, tag_to_index):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        super(BiLstmCrfNetwork, self).__init__()

        # Setup embedding variables
        self.tagset_size = len(tag_to_index)
        word_vectors = torch.tensor(wv.vectors)
        word_vectors = torch.cat((word_vectors, torch.zeros(1, wv.vector_size)))
        vector_size = wv.vector_size

        # Setup character embedding layers
        self.character_embeddings = nn.Embedding(len(string.printable) + 1, CHARACTER_EMBEDDING_SIZE, padding_idx=0)
        self.character_lstm = nn.LSTM(CHARACTER_EMBEDDING_SIZE, CHARACTER_HIDDEN_DIM, bidirectional=True)

        # Setup word embedding layer
        self.word_embeddings = nn.Embedding.from_pretrained(word_vectors)

        # The LSTM takes word embeddings concatenated with character verctors as inputs and
        # outputs hidden states with dimensionality hidden_dim.
        lstm_input_size = vector_size + CHARACTER_HIDDEN_DIM*2 + other_features
        self.lstm = nn.LSTM(lstm_input_size, HIDDEN_DIM, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(HIDDEN_DIM*2, self.tagset_size)

        self.crf = CRF(self.tagset_size)

    def _get_character_features(self, sentence):
        # Send each token through its own LSTM to get its character embeddings

        # Separate and pad character indices into a batch
        longest_token_length = max([len(token[1]) for token in sentence])
        character_indices = []
        for token in sentence:
            indices = [character for character in token[1]]
            if len(indices) < longest_token_length:
                padding = longest_token_length - len(indices)
                indices += [0] * padding
            character_indices.append(indices)
        character_indices = torch.tensor(character_indices)

        character_embeddings = self.character_embeddings(character_indices)
        character_embeddings = character_embeddings.permute(1, 0, 2)

        _, (hidden_output, _) = self.character_lstm(character_embeddings)
        character_vectors = hidden_output.permute(1, 0, 2).contiguous().view(-1, CHARACTER_HIDDEN_DIM*2)
        return character_vectors

    def _get_lstm_features(self, sentence):
        # Create tensor of word embeddings
        embedding_indices = [token[0] for token in sentence]
        embedding_indices = torch.tensor(embedding_indices, device=self.device)
        word_embeddings = self.word_embeddings(embedding_indices)

        character_vectors = self._get_character_features(sentence)

        # Turn rest of features into a tensor
        other_features = [token[2:] for token in sentence]
        other_features = torch.tensor(other_features, device=self.device)

        # Combine into one final input vector for LSTM
        token_vector = torch.cat((word_embeddings, character_vectors, other_features), 1)

        # Reshape because LSTM requires input of shape (seq_len, batch, input_size)
        token_vector = token_vector.view(len(sentence), 1, -1)
        # token_vector = self.dropout(token_vector)

        lstm_out, _ = self.lstm(token_vector)
        lstm_out = lstm_out.view(len(sentence), HIDDEN_DIM*2)

        lstm_features = self.hidden2tag(lstm_out)

        return lstm_features

    def forward(self, sentence):
        lstm_features = self._get_lstm_features(sentence)
        return lstm_features

class BiLstmCrfLearner:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    tag_to_index = {}
    untrained_tokens = set()
    other_features = []
    window_size = 0
    wv = None

    def __init__(self, word_embeddings):
        torch.manual_seed(1)
        self.character_to_index = {character:(index + 1) for index, character in enumerate(string.printable)}
        self.word_embeddings_file = word_embeddings

    def load_word_embeddings(self):
        if self.word_embeddings_file is None:
            raise ValueError('BiLSTM+CRF learner requires word embeddings.')

        if self.word_embeddings_file[-4] == '.bin':
            self.wv = KeyedVectors.load_word2vec_format(self.word_embeddings_file, binary=True)
        else:
            self.wv = KeyedVectors.load_word2vec_format(self.word_embeddings_file, binary=False)

    def find_other_features(self, example):
        if '0:text' not in example:
            raise ValueError('BiLSTM-CRF requires the "0:text" spaCy feature.')

        # Find other feature names
        for key in example:
            if key[:2] == '0:' and key != '0:text':
                self.other_features.append(key[2:])

    def devectorize_tag(self, tag_indices):
        to_tag = {y:x for x, y in self.tag_to_index.items()}
        tags = [to_tag[index] for index in tag_indices[0]]
        return tags

    def create_index_dictionary(self, sequences):
        to_index = {}
        for sequence in sequences:
            for item in sequence:
                if item not in to_index:
                    to_index[item] = len(to_index)
        return to_index

    def create_tag_dictionary(self, tags):
        tag_to_index = self.create_index_dictionary(tags)
        return tag_to_index

    def vectorize(self, sequence, to_index):
        # indices = [to_index[w] for w in sequence]
        indices = []

        for item in sequence:
            if item in to_index:
                indices.append(to_index[item])

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def find_window_size(self, x_data):
        """ Only supports single digit window sizes
        """
        test_token = None
        longest_length = 0
        for sentence in x_data:
            if len(sentence) > longest_length:
                longest_length = len(sentence)
                test_token = sentence[int(longest_length/2)]

        lowest = 0
        highest = 0

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

        return highest

    def find_window_indices(self, token):
        window = []
        window_range = range(-self.window_size, self.window_size + 1)

        for i in window_range:
            test_key = 'text'
            test_key = '%d:%s' % (i, test_key)
            if test_key in token:
                window.append(i)

        return window

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        s = re.sub(u"\u2013", "-", s) # em dash

        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in string.printable
        )
    
    def vectorize_tokens(self, tokens):
        tokens_vector = []

        for token in tokens:
            token_vector = []

            # Add text index for looking up word embedding
            token_text = token['0:text']
            token_text = self.unicodeToAscii(token_text)
            # p = re.compile(r'[A-z]*')
            # normalized_text = p.match(token_text).group()
            # normalized_text = ''.join(c.lower() for c in normalized_text)

            # Look up word embedding index
            # TODO Find correct way to handle this
            try:
                # embedding_index = self.wv[token_text]
                embedding_index = self.wv.vocab[token_text].index
            except KeyError:
                embedding_index = len(self.wv.vocab)

                # Only for logging untrained tokens
                # if normalized_text != '' and normalized_text != ' ' and not normalized_text.isnumeric():
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

        # return torch.tensor(tokens_vector, dtype=torch.long, device=self.device)
        return tokens_vector

    def fit(self, x_data, y_data):
        if self.wv is None:
            self.load_word_embeddings()

        self.tag_to_index = self.create_tag_dictionary(y_data)

        # Find other feature names
        self.find_other_features(x_data[0][0])

        # Calculate window size
        self.window_size = self.find_window_size(x_data)

        # Vectorize data
        sentences = []
        correct_tags = []

        for sentence, sentence_tags in zip(x_data, y_data):
            tokens_vector = self.vectorize_tokens(sentence)
            correct_tags_vector = self.vectorize(sentence_tags, self.tag_to_index)
            sentences.append(tokens_vector)
            correct_tags.append(correct_tags_vector)
        data = list(zip(sentences, correct_tags))

        other_features_length = len(self.other_features) * (self.window_size * 2 + 1)

        model = BiLstmCrfNetwork(self.wv, other_features_length, self.tag_to_index)

        if torch.cuda.is_available():
            logging.info('CUDA available. Moving model to GPU.')
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.NLLLoss()

        logging.info('Training BiLSTM...')

        for i in range(1, EPOCHS + 1):
            random.shuffle(data)
            epoch_losses = []
            for sentence, sentence_tags in data:
                # Training loop:

                if i < 21:
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
            logging.info('Epoch %d average loss: %f' % (i, average_loss))
            logging.debug(self.untrained_tokens)

        # for i, child in enumerate(model.children()):
        #     # Don't freeze CRF (child 5)
        #     if i < 5:
        #         for param in child.parameters():
        #             param.requires_grad = False

        # logging.info('Training CRF...')

        # for i in range(1, EPOCHS + 1):
        #     random.shuffle(data)
        #     epoch_losses = []
        #     for sentence, sentence_tags in data:
        #         # Training loop:
        #         emissions = model(sentence).unsqueeze(1)
        #         sentence_tags = sentence_tags.unsqueeze(1)
        #         loss = -model.crf(emissions, sentence_tags)

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         epoch_losses.append(loss)
        #     average_loss = sum(epoch_losses) / len(epoch_losses)
        #     logging.info('Epoch %d average loss: %f' % (i, average_loss))
        #     logging.debug(self.untrained_tokens)

        self.model = model

    def predict(self, sequences):
        if not self.wv:
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
        saved_data = torch.load(path)
        
        self.tag_to_index = saved_data['tag_to_index']
        self.other_features = saved_data['other_features']
        self.window_size = saved_data['window_size']

        model = saved_data['model']
        model.eval()
        self.model = model
        self.load_word_embeddings()
