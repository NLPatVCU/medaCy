"""Learner for running predictions and fine tuning BERT models.
"""
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

    def encode(self, sequence):
        encoded_sequence = []

        for token in sequence:
            encoded_sequence.append(self.tokenizer.encode(token))

        return self.tokenizer.build_inputs_with_special_tokens(encoded_sequence)

    def fit(self, x_data, y_data):
        self.vectorizer.create_tag_dictionary(y_data)

        self.model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(self.vectorizer.tag_to_index))
        self.model.train()
        self.model.to(device=self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        optimizer = Adam(self.model.parameters())

        encoded_sequences = [self.encode(sequence) for sequence in x_data]
        split_sequences = []
        split_sequence_labels = []

        for sequence, labels in zip(encoded_sequences, y_data):
            split_sequence = [sequence[0]]
            split_labels = ['O']

            for ids, label in zip(sequence[1:-1], labels):
                for token_id in ids:
                    split_sequence.append(token_id)
                    split_labels.append(label)

            split_sequence.append(sequence[-1])
            split_labels.append('O')

            split_sequences.append(split_sequence)
            split_sequence_labels.append(self.vectorizer.vectorize_tags(split_labels))

        for epoch in range(3):
            print('Epoch %d' % epoch)
            epoch_loss = 0.0

            for sequence, labels in zip(split_sequences, split_sequence_labels):
                sequence = torch.tensor(sequence, device=self.device).unsqueeze(0)
                labels = labels.unsqueeze(0)

                loss, _ = self.model(sequence, labels=labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            print(epoch_loss)

        self.model.save_pretrained('.')

    def predict(self, sequences):
        """Use model to make predictions over a given dataset.

        :param sequences: Sequences to predict labels for.
        :return: List of list of predicted labels.
        """
        predictions = []
        return predictions

    def load(self, path):
        self.model = BertForTokenClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
