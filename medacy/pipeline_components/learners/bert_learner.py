"""Learner for running predictions and fine tuning BERT models.
"""
import logging
import os
import torch
from torch.optim import Adam
from transformers import BertTokenizer, BertForTokenClassification

from medacy.nn import Vectorizer

class BertLearner:
    def __init__(self, cuda_device=-1):
        torch.manual_seed(1)
        device_string = 'cuda:%d' % cuda_device if cuda_device >= 0 else 'cpu'
        self.device = torch.device(device_string)

        self.model = None
        self.tokenizer = None
        self.vectorizer = Vectorizer(self.device)

    def encode_sequences(self, sequences, labels=[]):
        encoded_sequences = []

        if not labels:
            for sequence in sequences:
                labels.append(['O'] * len(sequence))

        for sequence in sequences:
            encoded_sequence = []
            for token in sequence:
                encoded_sequence.append(self.tokenizer.encode(token))
            encoded_sequence = self.tokenizer.build_inputs_with_special_tokens(encoded_sequence)
            encoded_sequences.append(encoded_sequence)

        split_sequences = []
        split_sequence_labels = []
        mappings = []

        for sequence, sequence_labels in zip(encoded_sequences, labels):
            split_sequence = [sequence[0]]
            split_labels = ['O']
            mapping = [0]

            for ids, label in zip(sequence[1:-1], sequence_labels):
                map_value = 1
                for token_id in ids:
                    split_sequence.append(token_id)
                    split_labels.append(label)
                    mapping.append(map_value)
                    map_value = 0

            split_sequence.append(sequence[-1])
            split_labels.append('O')
            mapping.append(0)

            split_sequences.append(split_sequence)
            split_sequence_labels.append(self.vectorizer.vectorize_tags(split_labels))
            mappings.append(mapping)

        return split_sequences, split_sequence_labels, mappings

    def decode_labels(self, sequence_labels, mappings):
        decoded_labels = []

        for labels, mapping in zip(sequence_labels, mappings):
            remapped_labels = []

            for label, map_value in zip(labels, mapping):
                if map_value == 1:
                    remapped_labels.append(label)

            decoded_labels.append(self.vectorizer.devectorize_tag(remapped_labels))

        return decoded_labels

    def fit(self, x_data, y_data):
        self.vectorizer.create_tag_dictionary(y_data)

        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-cased',
            num_labels=len(self.vectorizer.tag_to_index)
        )
        self.model.train()
        self.model.to(device=self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        optimizer = Adam(self.model.parameters())

        x_data, y_data, mappings = self.encode_sequences(x_data, y_data)
        
        for epoch in range(3):
            logging.info('Epoch %d' % epoch)

            for sequence, labels in zip(x_data, y_data):
                sequence = torch.tensor(sequence, device=self.device).unsqueeze(0)
                labels = labels.unsqueeze(0)

                loss, _ = self.model(sequence, labels=labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, sequences):
        """Use model to make predictions over a given dataset.

        :param sequences: Sequences to predict labels for.
        :return: List of list of predicted labels.
        """
        self.model.eval()
        encoded_sequences, _, mappings = self.encode_sequences(sequences)
        encoded_tag_indices = []
        predictions = []

        for sequence in encoded_sequences:
            sequence = torch.tensor(sequence, device=self.device).unsqueeze(0)
            scores = self.model(sequence)[0].squeeze()
            tag_indices = torch.max(scores, 1)[1].tolist()
            encoded_tag_indices.append(tag_indices)

        predictions = self.decode_labels(encoded_tag_indices, mappings)

        return predictions

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        vectorizer_values = self.vectorizer.get_values()
        torch.save(vectorizer_values, path + '/vectorizer.pt')

    def load(self, path):
        self.model = BertForTokenClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        vectorizer_values = torch.load(path + '/vectorizer.pt')
        self.vectorizer = Vectorizer(device=self.device)
        self.vectorizer.load_values(vectorizer_values)
