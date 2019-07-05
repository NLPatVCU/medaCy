"""
BiLSTM+CRF PyTorch network and model.

Original LSTM code was based off of tutorial found at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
CRF code based off of tutorial found at https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
"""
import logging
import random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Constants
LEARNING_RATE = 0.1
HIDDEN_DIM = 300
EPOCHS = 10
START_TAG = '<START>'
STOP_TAG = '<STOP>'

class BiLstmCrfNetwork(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, mimic_embeddings, other_features, tag_to_index):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        super(BiLstmCrfNetwork, self).__init__()

        self.tag_to_index = tag_to_index
        self.tagset_size = len(tag_to_index)

        self.word_embeddings = nn.Embedding.from_pretrained(mimic_embeddings)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        lstm_input_size = len(mimic_embeddings[0]) + other_features
        self.lstm = nn.LSTM(lstm_input_size, HIDDEN_DIM, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(HIDDEN_DIM*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_index[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_index[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = (
            torch.randn(2, 1, HIDDEN_DIM),
            torch.randn(2, 1, HIDDEN_DIM)
        )
        return hidden

    def argmax(self, vector):
        _, index = torch.max(vector, 1)
        return index.item()

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(self, vector):
        max_score = vector[0, self.argmax(vector)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vector.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vector - max_score_broadcast)))

    def _forward_alg(self, features):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_index[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feature in features:
            alphas_t = []  # The forward tensors at this timestep

            # Forward tensor + transition values + emissions scores (reshaped into a column instead of row)
            next_tag_var = forward_var + self.transitions + feature.view(self.tagset_size, -1)
            for row in next_tag_var:
                row = row.view(1, -1)
                log_sum = self.log_sum_exp(row).view(1)
                alphas_t.append(log_sum)

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_index[STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        embedding_indices = [token[0] for token in sentence]
        embedding_indices = torch.tensor(embedding_indices, device=self.device)

        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(embedding_indices)

        other_features = [token[1:] for token in sentence]
        other_features = torch.tensor(other_features, dtype=torch.float, device=self.device)
        embeds = torch.cat((embeds, other_features), 1)
        # Reshape because LSTM requires input of shape (seq_len, batch, input_size)
        embeds = embeds.view(len(sentence), 1, -1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), HIDDEN_DIM*2)
        lstm_features = self.hidden2tag(lstm_out)

        return lstm_features

    def _score_sentence(self, features, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_index[START_TAG]], dtype=torch.long, device=self.device), tags])
        for i, feature in enumerate(features):
            score += self.transitions[tags[i + 1], tags[i]] + feature[tags[i + 1]]
        score += self.transitions[self.tag_to_index[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_index[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_index[STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_index[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        features = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(features)
        gold_score = self._score_sentence(features, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_features = self._get_lstm_features(sentence)

        # tag_scores = F.log_softmax(lstm_features, dim=1)
        score, tag_seq = self._viterbi_decode(lstm_features)

        # return tag_scores
        return score, tag_seq

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

        mimic_embeddings = torch.FloatTensor(mimic_embeddings, device=self.device)
        self.token_to_index = token_to_index
        self.mimic_embeddings = mimic_embeddings

    def devectorize_tag(self, tag_indices):
        to_tag = {y:x for x, y in self.tag_to_index.items()}
        tags = [to_tag[index] for index in tag_indices]
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
        tag_to_index[START_TAG] = len(tag_to_index)
        tag_to_index[STOP_TAG] = len(tag_to_index)
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
            test_key = self.other_features[0]
            test_key = '%d:%s' % (i, test_key)
            if test_key in token:
                window.append(i)

        return window

    def vectorize_tokens(self, tokens):
        tokens_vector = []

        for token in tokens:
            token_vector = []

            # Add text index for looking up word embedding
            token_text = token['0:norm_']
            token_text = ''.join(c for c in token_text if c.isalpha())

            # TODO Find correct way to handle this
            if token_text not in self.token_to_index:
                if token_text != '' and token_text != ' ' and not token_text.isnumeric():
                    self.untrained_tokens.add(token_text)
                embedding_index = self.token_to_index['UNTRAINED']
            else:
                embedding_index = self.token_to_index[token_text]

            token_vector.append(embedding_index)

            # Find window indices
            window = self.find_window_indices(token)

            # Add features to vector in order
            for i in window:
                for feature in self.other_features:
                    key = '%d:%s' % (i, feature)
                    feature = float(token[key])
                    token_vector.append(feature)

            # Pad vector when the sequence is shorter than the window size
            # Features * possible indices + 1 for embedding index
            expected_length = len(self.other_features) * (self.window_size * 2 + 1) + 1
            if len(token_vector) < expected_length:
                missing_values = expected_length - len(token_vector)
                token_vector.extend([float(0) for _ in range(missing_values)])

            tokens_vector.append(token_vector)

        # return torch.tensor(tokens_vector, dtype=torch.long, device=self.device)
        return tokens_vector

    def fit(self, x_data, y_data):
        self.tag_to_index = self.create_tag_dictionary(y_data)

        # Find other feature names
        for key in x_data[0][0]:
            if key[:2] == '0:' and key != '0:norm_':
                self.other_features.append(key[2:])

        # Calculate window size
        self.window_size = self.find_window_size(x_data)

        other_features_length = len(self.other_features) * (self.window_size * 2 + 1)

        model = BiLstmCrfNetwork(self.mimic_embeddings, other_features_length, self.tag_to_index)

        if torch.cuda.is_available():
            logging.info('GPU available. Moving model to CUDA.')
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

                # Run prediction
                # prediction_scores = model(tokens_vector)

                # Compute loss and train network based on it
                loss = model.neg_log_likelihood(tokens_vector, correct_tags_vector)

                # loss.backward()
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
                _, tag_indices = self.model(vectorized_tokens)
                predictions.append(self.devectorize_tag(tag_indices))

        return predictions

    def save(self, path):
        properties = {
            'model': self.model,
            'token_to_index': self.token_to_index,
            'tag_to_index': self.tag_to_index
        }

        if path[-4:] != '.pth':
            path += '.pth'

        torch.save(properties, path)

    def load(self, path):
        saved_data = torch.load(path)
        
        self.token_to_index = saved_data['token_to_index']
        self.tag_to_index = saved_data['tag_to_index']

        model = saved_data['model']
        model.eval()
        self.model = model
