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

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

# Constants
LEARNING_RATE = 0.01
HIDDEN_DIM = 300
CHARACTER_HIDDEN_DIM = 100
EPOCHS = 10

class BiLstmCrfNetwork(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, mimic_embeddings, other_features, tag_to_index):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        super(BiLstmCrfNetwork, self).__init__()

        # Setup character embedding layers
        self.character_embeddings = nn.Embedding(len(string.printable), CHARACTER_HIDDEN_DIM)
        self.character_lstm = nn.LSTM(CHARACTER_HIDDEN_DIM, CHARACTER_HIDDEN_DIM, bidirectional=True)

        # Setup word embedding layers
        self.tagset_size = len(tag_to_index)
        self.word_embeddings = nn.Embedding.from_pretrained(mimic_embeddings)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        lstm_input_size = len(mimic_embeddings[0]) + CHARACTER_HIDDEN_DIM*2 + other_features
        self.lstm = nn.LSTM(lstm_input_size, HIDDEN_DIM, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(HIDDEN_DIM*2, self.tagset_size)

        self.crf = CRF(self.tagset_size)
        

    def _get_lstm_features(self, sentence):
        # Create tensor of word embeddings
        embedding_indices = [token[0] for token in sentence]
        embedding_indices = torch.tensor(embedding_indices, device=self.device)
        word_embeddings = self.word_embeddings(embedding_indices)

        # Send each token through its own LSTM to get its character embeddings
        character_vectors = []
        for token in sentence:
            character_indices = [character for character in token[1]]
            character_indices = torch.tensor(character_indices, device=self.device).unsqueeze(0)
            character_embeddings = self.character_embeddings(character_indices).view(len(token[1]), 1, -1)
            _, (h_n, _) = self.character_lstm(character_embeddings)
            character_vector = h_n.view(1, CHARACTER_HIDDEN_DIM*2)
            character_vectors.append(character_vector)
        character_vectors = torch.cat(character_vectors)

        # Turn rest of features into a tensor
        other_features = [token[2:] for token in sentence]
        other_features = torch.tensor(other_features, device=self.device)

        # Combine into one final input vector for LSTM
        token_vector = torch.cat((word_embeddings, character_vectors, other_features), 1)
        # Reshape because LSTM requires input of shape (seq_len, batch, input_size)
        token_vector = token_vector.view(len(sentence), 1, -1)

        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = self.lstm(token_vector)
        lstm_out = lstm_out.view(len(sentence), HIDDEN_DIM*2)
        # lstm_out = torch.cat((lstm_out, other_features), 1)

        lstm_features = self.hidden2tag(lstm_out)

        return lstm_features

    def forward(self, sentence):
        lstm_features = self._get_lstm_features(sentence)
        return lstm_features

class BiLstmCrfLearner:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    token_to_index = {}
    tag_to_index = {}
    mimic_embeddings = []
    untrained_tokens = set()
    other_features = []
    window_size = 0

    def __init__(self, word_embeddings):
        torch.manual_seed(1)
        self.character_to_index = {character:index for index, character in enumerate(string.printable)}
        self.load_word_embeddings(word_embeddings)

    def load_word_embeddings(self, word_embeddings):
        if word_embeddings is None:
            raise ValueError('BiLSTM+CRF learner requires word embeddings.')

        logging.info('Preparing mimic word embeddings...')

        with open(word_embeddings) as mimic_file:
            token_to_index = {}
            mimic_embeddings = []

            # Read first line so it's not included in the loop
            mimic_file.readline()

            for line in mimic_file:
                values = line.split(' ')

                token = values[0]

                embeddings = values[1:]
                if '\n' in embeddings:
                    embeddings.remove('\n')
                embeddings = list(map(float, embeddings))

                token_to_index[token] = len(token_to_index)
                mimic_embeddings.append(embeddings)

        mimic_embeddings.append([float(1) for _ in range(200)])
        token_to_index['UNTRAINED'] = len(token_to_index)

        mimic_embeddings = torch.tensor(mimic_embeddings, device=self.device)
        self.token_to_index = token_to_index
        self.mimic_embeddings = mimic_embeddings

    def find_other_features(self, example):
        contains_text = False
        # Find other feature names
        for key in example:
            if key == '0:text':
                contains_text = True
            elif key[:2] == '0:':
                self.other_features.append(key[2:])

        if not contains_text:
            raise ValueError('BiLSTM-CRF requires the "0:text" spaCy feature.')

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
            test_key = '0:norm_'
            test_key = '%d:%s' % (i, test_key)
            if test_key in token:
                window.append(i)

        return window

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
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
            normalized_text = ''.join(c.lower() for c in token_text if c.isalpha())

            # Look up word embedding index
            # TODO Find correct way to handle this
            if normalized_text not in self.token_to_index:
                embedding_index = self.token_to_index['UNTRAINED']

                # Only for logging untrained tokens
                if normalized_text != '' and normalized_text != ' ' and not normalized_text.isnumeric():
                    self.untrained_tokens.add(token_text)
                
            else:
                embedding_index = self.token_to_index[normalized_text]

            token_vector.append(embedding_index)

            # Add list of character indices as second item
            character_indices = []
            for character in token_text:
                ascii_character = self.unicodeToAscii(character)
                index = self.character_to_index[ascii_character]
                character_indices.append(index)
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
        self.tag_to_index = self.create_tag_dictionary(y_data)

        # Find other feature names
        self.find_other_features(x_data[0][0])

        # Calculate window size
        self.window_size = self.find_window_size(x_data)

        other_features_length = len(self.other_features) * (self.window_size * 2 + 1)

        model = BiLstmCrfNetwork(self.mimic_embeddings, other_features_length, self.tag_to_index)

        if torch.cuda.is_available():
            logging.info('CUDA available. Moving model to GPU.')
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        for i in range(EPOCHS):
            epoch_losses = []
            for tokens, correct_tags in zip(x_data, y_data):
                # Reset optimizer weights
                optimizer.zero_grad()

                # Vectorize input and test data
                tokens_vector = self.vectorize_tokens(tokens)
                correct_tags_vector = self.vectorize(correct_tags, self.tag_to_index)

                # Training loop:
                prediction = model(tokens_vector).unsqueeze(1)
                correct_tags_vector = correct_tags_vector.unsqueeze(1)
                loss = -model.crf(prediction, correct_tags_vector)

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)
            average_loss = sum(epoch_losses) / len(epoch_losses)
            logging.info('Epoch %d average loss: %f' % (i, average_loss))
            logging.debug(self.untrained_tokens)

        self.model = model

    def predict(self, sequences):
        if not self.token_to_index:
            raise RuntimeError('There is no token_to_index. Model must not have been fit yet'
                'or was loaded improperly.')
        
        with torch.no_grad():
            predictions = []
            for sequence in sequences:
                vectorized_tokens = self.vectorize_tokens(sequence)
                lstm_features = self.model(vectorized_tokens).unsqueeze(1)
                tag_indices = self.model.crf.decode(lstm_features)
                predictions.append(self.devectorize_tag(tag_indices))

        return predictions

    def save(self, path):
        properties = {
            'model': self.model,
            'token_to_index': self.token_to_index,
            'tag_to_index': self.tag_to_index,
            'other_features': self.other_features,
            'window_size': self.window_size
        }

        if path[-4:] != '.pth':
            path += '.pth'

        torch.save(properties, path)

    def load(self, path):
        saved_data = torch.load(path)
        
        self.token_to_index = saved_data['token_to_index']
        self.tag_to_index = saved_data['tag_to_index']
        self.other_features = saved_data['other_features']
        self.window_size = saved_data['window_size']

        model = saved_data['model']
        model.eval()
        self.model = model
