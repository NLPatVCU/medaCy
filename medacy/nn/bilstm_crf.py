"""
BiLSTM-CRF Pytorch Network
"""
import string

import torch
import torch.nn as nn
from torchcrf import CRF

HIDDEN_DIM = 200
CHARACTER_HIDDEN_DIM = 100
CHARACTER_EMBEDDING_SIZE = 100

class BiLstmCrf(nn.Module):
    """
    BiLSTM and CRF pytorch layers.

    :ivar device: Pytorch device.
    :ivar tagset_size: Number of labels that the network is being trained for.
    """
    def __init__(self, word_vectors, other_features, tag_to_index, device):
        """Init model.

        :param word_vectors: Gensim word vector object to use as word embeddings.
        :param other_features: Number of other word features being used.
        :param tag_to_index: Dictionary for mapping tag/label to an index for vectorization.
        :param device: Pytorch device to use.
        """
        self.device = device

        if device.type != 'cpu':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        super(BiLstmCrf, self).__init__()

        # Setup embedding variables
        self.tagset_size = len(tag_to_index)
        vector_size = word_vectors.vector_size
        word_vectors = torch.tensor(word_vectors.vectors, device=device)
        word_vectors = torch.cat((word_vectors, torch.zeros(1, vector_size)))

        # Setup character embedding layers
        self.character_embeddings = nn.Embedding(
            len(string.printable) + 1,
            CHARACTER_EMBEDDING_SIZE,
            padding_idx=0
        )

        self.character_lstm = nn.LSTM(
            CHARACTER_EMBEDDING_SIZE,
            CHARACTER_HIDDEN_DIM,
            bidirectional=True
        )

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
        """Send each token through its own LSTM to get its character embeddings

        :param sentence: List of string tokens.
        :return: List character LSTM hidden state outputs.
        """

        # Separate and pad character indices into a batch
        longest_token_length = max([len(token[1]) for token in sentence])
        character_indices = []
        for token in sentence:
            indices = [character for character in token[1]]
            if len(indices) < longest_token_length:
                padding = longest_token_length - len(indices)
                indices += [0] * padding
            character_indices.append(indices)
        character_indices = torch.tensor(character_indices, device=self.device)

        # Get character embeddings based on indices
        character_embeddings = self.character_embeddings(character_indices)
        character_embeddings = character_embeddings.permute(1, 0, 2)

        # Run embeddings through character BiLSTM
        _, (hidden_output, _) = self.character_lstm(character_embeddings)
        features = hidden_output.permute(1, 0, 2).contiguous().view(-1, CHARACTER_HIDDEN_DIM*2)

        return features

    def _get_lstm_features(self, sentence):
        """Get BiLSTM features from a list of tokens

        :param sentence: List of string tokens.
        :return: Output from BiLSTM.
        """
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
