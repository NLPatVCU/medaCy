
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import torch
from torch import nn


class LstmDataset(Dataset):

    tokens = None
    chars = None
    tags = None
    masks = None
    features = None

    def __init__(self, data, embeddings_file, device):
        sentences, sentence_tags = zip(*data)
        self.X_data = sentences
        self.y_data = sentence_tags
        binary = False
        if embeddings_file.endswith('.bin'):
            binary = True
        self.word_vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=binary)
        word_vectors = torch.tensor(self.word_vectors.vectors, device=device)
        word_vectors = torch.cat((word_vectors, torch.zeros(1, self.word_vectors.vector_size, device=device)))
        self.token_embed = nn.Embedding.from_pretrained(word_vectors)
        self.encode()

    def __getitem__(self, item):
        return [{'token': self.tokens[item], 'char': self.chars[item],
                 'other_features': self.features[item]}, {'tags': self.tags[item], 'mask': self.masks[item]}]

    def __len__(self):
        return len(self.X_data)

    def encode(self, max_token_length=50, max_char_length=20):
        self.load_tokens(max_token_length)
        self.load_chars(max_token_length, max_char_length)
        self.load_other_features(max_token_length)
        self.load_tags(max_token_length)

    def load_tokens(self, token_length):
        padding_token = len(self.word_vectors.vocab)
        token_indices = torch.zeros((len(self.X_data), token_length), dtype=torch.long)
        for i, sequence in enumerate(self.X_data):
            tokens = []
            for token in sequence:
                tokens.append(token[0])
            if len(tokens) > token_length:
                tokens = tokens[:token_length]
            else:
                tokens += [padding_token] * (token_length - len(tokens))
            tokens = torch.tensor(tokens, dtype=torch.long)
            token_indices[i] = tokens
        self.tokens = self.token_embed(token_indices)

    def load_chars(self, token_length, char_length):
        sequence_padding = [0] * char_length
        char_indices = torch.zeros((len(self.X_data), token_length, char_length), dtype=torch.long)
        for i, sequence in enumerate(self.X_data):
            sequence_indices = []
            for j, token in enumerate(sequence):
                indices = token[1]
                if len(indices) > char_length:
                    indices = indices[:char_length]
                else:
                    indices += [0] * (char_length - len(indices))
                sequence_indices.append(indices)
            if len(sequence_indices) > token_length:
                sequence_indices = sequence_indices[:token_length]
            else:
                sequence_indices += [sequence_padding] * (token_length - len(sequence_indices))
            sequence_tensor = torch.tensor(sequence_indices, dtype=torch.long)
            char_indices[i] = sequence_tensor
        self.chars = char_indices

    def load_other_features(self, token_length):
        feature_length = len(self.X_data[0][0][2:])
        feature_pad = [0] * feature_length
        other_features = torch.zeros((len(self.X_data), token_length, feature_length))
        for i, sequence in enumerate(self.X_data):
            sequence_features = []
            for token in sequence:
                features = token[2:]
                sequence_features.append(features)
            if len(sequence_features) > token_length:
                sequence_features = sequence_features[:token_length]
            else:
                sequence_features += [feature_pad] * (token_length - len(sequence_features))
            features_tensor = torch.tensor(sequence_features)
            other_features[i] = features_tensor
        self.features = other_features

    def load_tags(self, token_length):
        tags = torch.zeros((len(self.X_data), token_length), dtype=torch.long)
        masks = torch.zeros((len(self.X_data), token_length), dtype=torch.uint8)
        for i, sequence in enumerate(self.y_data):
            sequence = list(sequence)
            if len(sequence) > token_length:
                mask = [1] * token_length
                sequence = sequence[:token_length]
            else:
                mask = [1] * len(sequence) + [0] * (token_length - len(sequence))
                sequence += [-1] * (token_length - len(sequence))
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)
            mask_tensor = torch.tensor(mask, dtype=torch.long)
            tags[i] = sequence_tensor
            masks[i] = mask_tensor
        self.tags = tags
        self.masks = masks
