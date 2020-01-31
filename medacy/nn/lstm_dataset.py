
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from torch.nn import Embedding
import torch


class LstmDataset(Dataset):

    x_data = None
    y_data = None
    masks = []

    def __init__(self, data, embeddings_file, device):
        sequences, labels = zip(*data)
        self.max_seq_len = max([len(seq) for seq in sequences])
        self.device = device

        self.embed = None
        self.token_to_index = {}
        self.tag_to_index = {}
        self.padding_index = None

        self.load_embeddings(embeddings_file)
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
            tokens = [token['0:text'] for token in sequence]
            indices = []
            for token in tokens:
                try:
                    indices.append(self.token_to_index[token])
                except KeyError:
                    indices.append(self.padding_index)
            if len(indices) < self.max_seq_len:
                pad = [self.padding_index] * (self.max_seq_len - len(indices))
                indices += pad
            all_indices[i] = (torch.tensor(indices, device=self.device))

        embedded = self.embed(all_indices)
        self.x_data = embedded

    def vectorize_labels(self, labels):
        self.create_tag_dictionary(labels)
        all_labels = torch.zeros((len(labels), self.max_seq_len), device=self.device, dtype=torch.long)
        all_masks = torch.zeros((len(labels), self.max_seq_len), device=self.device, dtype=torch.long)
        for i, label in enumerate(labels):
            tags = [self.tag_to_index[token] for token in label]
            mask = [1] * len(tags)
            if len(tags) < self.max_seq_len:
                pad = [0] * (self.max_seq_len - len(tags))
                tags += pad
                mask += pad
            all_masks[i] = torch.tensor(mask, device=self.device)
            all_labels[i] = torch.tensor(tags, device=self.device)
        self.y_data = all_labels
        self.masks = all_masks

    def create_tag_dictionary(self, tags):
        tag_to_index = {}
        for sequence in tags:
            for tag in sequence:
                if tag not in tag_to_index:
                    tag_to_index[tag] = len(tag_to_index)
        self.tag_to_index = tag_to_index

    def load_embeddings(self, embeddings_file):
        is_binary = embeddings_file.endswith('.bin')
        word_vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=is_binary)

        index_to_tag = word_vectors.index2word
        self.token_to_index = {tag: i for i, tag in enumerate(index_to_tag)}
        vectors = torch.tensor(word_vectors.wv.vectors, device=self.device)
        padding = torch.zeros((1, word_vectors.vector_size), device=self.device)
        vectors = torch.cat((vectors, padding), dim=0)
        self.embed = Embedding.from_pretrained(vectors, padding_idx=vectors.shape[0] - 1)
        self.padding_index = vectors.shape[0] - 1
