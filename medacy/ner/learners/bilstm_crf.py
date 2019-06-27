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
EMBEDDING_DIM = 50
HIDDEN_DIM = 300
EPOCHS = 10

START_TAG = '<START>'
STOP_TAG = '<STOP>'

class BiLstmCrfNetwork(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, vocab_size, tag_to_index):
        super(BiLstmCrfNetwork, self).__init__()

        self.tag_to_index = tag_to_index
        self.tagset_size = len(tag_to_index)

        self.word_embeddings = nn.Embedding(vocab_size, EMBEDDING_DIM)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, bidirectional=True)

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

            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feature[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_index[STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)
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

    # def forward(self, sentence):
    #     embeds = self.word_embeddings(sentence)
    #     lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    #     tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    #     tag_scores = F.log_softmax(tag_space, dim=1)
    #     return tag_scores

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

    def __init__(self):
        torch.manual_seed(1)

    def devectorize_tags(self, tags_vectors):
        to_tag = {y:x for x, y in self.tag_to_index.items()}
        tags = []

        for vector in tags_vectors:
            max_value = max(vector)
            index = list(vector).index(max_value)
            tags.append(to_tag[index])
            
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
            else: # TODO Only here for testing until we switch to word embeddings
                indices.append(random.randrange(len(to_index)))

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def fit(self, x_data, y_data):
        self.token_to_index = self.create_index_dictionary(x_data)
        self.tag_to_index = self.create_tag_dictionary(y_data)

        vocab_size = len(self.token_to_index)
        model = BiLstmCrfNetwork(vocab_size, self.tag_to_index)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        for i in range(EPOCHS):
            epoch_losses = []
            for tokens, correct_tags in zip(x_data, y_data):
                # Reset optimizer weights
                optimizer.zero_grad()

                # Vectorize input and test data
                tokens_vector = self.vectorize(tokens, self.token_to_index)
                correct_tags_vector = self.vectorize(correct_tags, self.tag_to_index)

                # Run prediction
                prediction_scores = model(tokens_vector)

                # Compute loss and train network based on it
                # loss = loss_function(prediction_scores, correct_tags_vector)
                loss = model.neg_log_likelihood(tokens_vector, correct_tags_vector)

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)
            average_loss = sum(epoch_losses) / len(epoch_losses)
            logging.info('Epoch %d average loss: %f' % (i, average_loss))


        self.model = model

    def predict(self, sequences):
        if not self.token_to_index:
            raise RuntimeError('There is no token_to_index. Model must not have been fit yet'
                'or was loaded improperly.')
        
        with torch.no_grad():
            predictions = []
            for sequence in sequences:
                vectorized_tokens = self.vectorize(sequence, self.token_to_index)
                tag_scores = self.model(vectorized_tokens)
                predictions.append(self.devectorize_tags(tag_scores))
            # predictions.append(devectorized_tags)

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
