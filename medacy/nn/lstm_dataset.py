
from torch.utils.data import Dataset
from torch.nn import Embedding
import torch


class LstmDataset(Dataset):

    x_data = None
    y_data = None
    masks = []

    def __init__(self, data, word_vectors, device, tag_to_index):
        sequences, labels = zip(*data)
        self.max_seq_len = max([len(seq) for seq in sequences])
        self.device = device
        self.tag_to_index = tag_to_index

        self.embed = None
        self.token_to_index = {}
        self.tag_to_index = {}
        self.padding_index = None

        vectors = torch.tensor(word_vectors.vectors, device=self.device)
        padding = torch.zeros((1, vectors.shape[1]), device=self.device)
        vectors = torch.cat((vectors, padding), dim=0)
        self.embed = Embedding.from_pretrained(vectors, padding_idx=vectors.shape[0] - 1)
        self.padding_index = vectors.shape[0] - 1
        self.encode(sequences, labels)

    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, i):
        return {'sequence': self.x_data[i], 'mask': self.masks[i], 'labels': self.y_data[i]}

    def encode(self, sequences, labels):
        self.vectorize_sequences(sequences)
        self.vectorize_labels(labels)

    def vectorize_sequences(self, sequences):
        all_indices = torch.zeros((len(sequences), self.max_seq_len), device=self.device, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            token_indices = [token[0] for token in sequence]
            if len(token_indices) < self.max_seq_len:
                pad = [self.padding_index] * (self.max_seq_len - len(token_indices))
                token_indices += pad
            token_tensor = torch.tensor(token_indices, device=self.device)
            all_indices[i] = token_tensor

        embedded = self.embed(all_indices)
        self.x_data = embedded

    def vectorize_labels(self, labels):
        all_labels = torch.zeros((len(labels), self.max_seq_len), device=self.device, dtype=torch.long)
        all_masks = torch.zeros((len(labels), self.max_seq_len), device=self.device, dtype=torch.long)
        for i, tags in enumerate(labels):
            tags = tags.to(torch.uint8)
            mask = torch.ones(len(tags), device=self.device, dtype=torch.uint8)
            if tags.shape[0] < self.max_seq_len:
                pad = torch.zeros((self.max_seq_len - len(tags)), device=self.device, dtype=torch.uint8)
                tags = torch.cat((tags, pad))
                mask = torch.cat((mask, pad))
            all_masks[i] = mask
            all_labels[i] = tags
        self.y_data = all_labels
        self.masks = all_masks

