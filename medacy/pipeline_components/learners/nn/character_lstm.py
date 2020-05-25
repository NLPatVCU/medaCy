"""
BiLSTM-CRF PyTorch Network
"""
import string

import torch.nn as nn

class CharacterLSTM(nn.Module):
    """
    BiLSTM and CRF pytorch layers.

    :ivar device: PyTorch device.
    """
    def __init__(self, embedding_dim=100, padding_idx=0, hidden_size=100):

        super(CharacterLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.character_embeddings = nn.Embedding(
            num_embeddings=len(string.printable) + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.character_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, indices):
        # Get character embeddings based on indices
        character_embeddings = self.character_embeddings(indices)

        # Run embeddings through character BiLSTM
        _, (hidden_output, _) = self.character_lstm(character_embeddings)
        features = hidden_output.transpose(0, 1)
        features = features.contiguous().view(-1, self.hidden_size*2)

        return features
