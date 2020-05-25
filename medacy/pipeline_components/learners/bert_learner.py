"""
Learner for training and predicting with a BERT model.
"""
import logging
import os

import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import AdamW, BertTokenizer, BertForTokenClassification

from medacy.pipeline_components.learners.nn.bert_crf_for_token_classification import BertCrfForTokenClassification
from medacy.pipeline_components.learners.nn.sequences_dataset import SequencesDataset
from medacy.pipeline_components.learners.nn.vectorizer import Vectorizer


class BertLearner:
    """Learner for running predictions and fine tuning BERT models."""

    def __init__(
            self,
            cuda_device=-1,
            pretrained_model='bert-large-cased',
            batch_size=8,
            learning_rate=1e-5,
            epochs=3,
            using_crf=False):
        """
        Initialize BertLearner.

        :param cuda_device: CUDA device to use when running on a GPU. Use -1 for CPU.
        :param pretrained_model: Name of model to use for the Transformers pretrained_model. This
            is different from the medaCy fine-tuned model you may have created.
        :param batch_size: Size of each batch during training or cross validation.
        :param learning_rate: Learning rate to use for PyTorch optimizier
        :param epochs: Number of epochs to train for.
        :param using_crf: Whether or not to use a CRF layer on the model.
        """
        torch.manual_seed(1)  # Seed PyTorch for consistency

        # Create torch.device that all tensors will be using
        device_string = f'cuda:%d' % cuda_device if cuda_device >= 0 else 'cpu'
        self.device = torch.device(device_string)

        self.model = None  # Transformers/PyTorch model
        self.tokenizer = None  # Transformers tokenizer
        self.vectorizer = Vectorizer(self.device)  # medaCy vectorizer
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.using_crf = using_crf

    def encode_sequences(self, sequences, labels=None):
        """
        Encode a list of text sequences into encoded sequences for the Transformers package

        :param sequences: List of sequences. Each sequence is a list of strings that represent
            tokens from the original sentence.
        :param labels: List of list of labels. There's one label for each token in sequences.
            Labels are represented by strings.
        """
        encoded_sequences = []

        # If there are no labels then make all of the labels 'O' so they can still be used for
        # mapping during predictions
        if not labels:
            labels = [['O'] * len(sequence) for sequence in sequences]

        # Use the tokenizer to encode every sequence and add in special tokens like [CLS] and [SEP]
        # Then add them to encoded_sequences
        for sequence in sequences:
            encoded_sequence = [self.tokenizer.encode(token, add_special_tokens=False) for token in sequence]
            encoded_sequence = self.tokenizer.build_inputs_with_special_tokens(encoded_sequence)
            encoded_sequences.append(encoded_sequence)

        # Encoding them this way makes it possible to track which tokens were split up
        split_sequences = []
        split_sequence_labels = []

        # Iterate through encoded sequences and original labels. Each token in the sequnece is a
        # list that will be length 1 if the token wasn't split, but longer if it was
        for sequence, sequence_labels in zip(encoded_sequences, labels):
            # First token is always encoding of [CLS]. Create new lists starting with it and the
            # proper label/mapping
            split_sequence = [sequence[0]]
            split_labels = ['O']

            # Loop through ids and labels in sequences. Don't take first or last id lists from
            # sequence since we already added [CLS] and will add [SEP] later.
            for ids, label in zip(sequence[1:-1], sequence_labels):
                # Depending on the pretrained model, some tokens may be removed. Replace them
                # with unknown token id instead of removing
                if not ids:
                    ids = [self.tokenizer.unk_token_id]

                # Always add first id (which is the only id if the word was not split)
                split_sequence.append(ids[0])
                split_labels.append(label)

                # If the token is split, loop through the rest of the ids
                for token_id in ids[1:]:
                    # Add proper token id and label to lists
                    split_sequence.append(token_id)
                    split_labels.append('X')  # 'X' marks a token we should ignore

            # Add the final token [SEP] with proper label/mapping
            split_sequence.append(sequence[-1])
            split_labels.append('O')

            # Append final forms of lists
            split_sequences.append(split_sequence)
            split_sequence_labels.append(self.vectorizer.vectorize_tags(split_labels))

        # Return the two lists that were created. Note that when no labels were supplied
        # split_sequence_labels is only used to mark which tokens to predict for (mappings)
        return split_sequences, split_sequence_labels

    def decode_labels(self, sequence_labels, mappings):
        """
        Decode list of label indices using mappings generated during self.encode_sequences()

        :param sequence_labels: List of list of label indices. Index corresponds to index of label
            in self.vectorizer.
        :param mappings: List of list of mappings. All mappings indicate whether we should include
            it in classification (0) or not (null_label id)
        """
        decoded_labels = []
        null_label = self.vectorizer.tag_to_index['X']

        for labels, mapping in zip(sequence_labels, mappings):
            remapped_labels = []

            # Loop through labels and maps not including first and last items which are the
            # special tokens [CLS] and [SEP]
            for label, map_value in zip(labels[1:-1], mapping[1:-1]):
                # Only include label if it was not given the null label ('X') during encoding
                if map_value != null_label:
                    remapped_labels.append(label)

            # Decode list of label indices useing self.vectorizer
            decoded_labels.append(self.vectorizer.devectorize_tag(remapped_labels))

        return decoded_labels

    def fit(self, x_data, y_data):
        """Finetune a pretrained BERT model using the Transformers package.

        :param x_data: List of sequences. Each sequence is a list of strings that represent
            tokens from the original sentence.
        :param y_data: List of list of labels. There's one label for each token in sequences.
            Labels are represented by strings.
        """
        # Prepare label encodings
        self.vectorizer.create_tag_dictionary(y_data)
        self.vectorizer.add_tag('X')

        # Decide on model class based on whether we're using a CRF layer or not
        model_class = BertCrfForTokenClassification if self.using_crf else BertForTokenClassification

        # Load pretrained BERT model, unfreeze layers, move it to GPU device, and create its tokenizer
        self.model = model_class.from_pretrained(
            self.pretrained_model,
            num_labels=len(self.vectorizer.tag_to_index) - 1  # Don't include 'X'
        )
        self.model.train()
        self.model.to(device=self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)

        # Use Adam optimizer with weight decay options as shown in the link below:
        # https://github.com/huggingface/transformers/blob/master/examples/run_ner.py
        # Currently not using any decay, but left this here for future reference.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n,
                    p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            },
            {
                "params": [
                    p for n,
                    p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)

        # Encode sequences and labels and prepare dataset/loader
        sequences, labels = self.encode_sequences(x_data, y_data)
        dataset = SequencesDataset(
            device=self.device,
            sequences=sequences,
            x_label=self.vectorizer.tag_to_index['X'],
            sequence_labels=labels,
            o_label=self.vectorizer.tag_to_index['O']
        )
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=dataset.collate
        )

        # Training loop
        for epoch in range(self.epochs):
            logging.info(f'Epoch {epoch}')
            training_loss = 0
            batches = 0

            for sequences, masks, labels in dataloader:
                # Pass sequences through the model and get the loss
                loss, _ = self.model(sequences, labels=labels, attention_mask=masks)

                # Train using the optimizer and loss
                loss.backward()
                training_loss += loss.item()
                batches += 1
                optimizer.step()
                self.model.zero_grad()

            logging.info(f'Loss: {training_loss / batches}')

    def predict(self, x_data):
        """Use model to make predictions over a given dataset.

        :param x_data: Sequences to predict labels for.
        :return: List of list of predicted labels.
        """
        if self.model is None:
            raise ValueError('Please load or train model first.')

        # Freeze model weights
        self.model.eval()

        # Encode sequences and prepare dataset/loader
        encoded_sequences, mappings = self.encode_sequences(x_data, labels=[])
        encoded_tag_indices = []
        dataset = SequencesDataset(
            device=self.device,
            sequences=encoded_sequences,
            sequence_labels=mappings,
            o_label=self.vectorizer.tag_to_index['O'],
            x_label=self.vectorizer.tag_to_index['X'],
        )
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=dataset.collate
        )

        # Loop through batches to make predictions
        for batch in dataloader:
            sequences, attention_masks, _ = batch

            # Get emission scores
            scores = self.model(sequences, attention_mask=attention_masks)[0]

            if self.using_crf:
                # Use pytorch-crf package to do a Viterbi decode
                tag_indices = self.model.crf.decode(scores)
                encoded_tag_indices.extend(tag_indices)
            else:
                # Predict label with max emission score for each token
                tag_indices = torch.max(scores, 2)[1].tolist()
                encoded_tag_indices.extend(tag_indices)

        # Decode and return final label predictions
        predictions = self.decode_labels(encoded_tag_indices, mappings)
        return predictions

    def save(self, path):
        """Save trained model and vectorizer.

        :param path: Path of directory to save model in.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        vectorizer_values = self.vectorizer.get_values()
        torch.save(vectorizer_values, path + '/vectorizer.pt')

    def load(self, path):
        """Load saved model and vectorizer.

        :param path: Path of directory where the model was saved.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        vectorizer_values = torch.load(path + '/vectorizer.pt')
        self.vectorizer = Vectorizer(device=self.device)
        self.vectorizer.load_values(vectorizer_values)
        model_class = BertCrfForTokenClassification if self.using_crf else BertForTokenClassification
        self.model = model_class.from_pretrained(
            path,
            num_labels=len(self.vectorizer.tag_to_index) - 1  # Ignore 'X'
        )
        self.model = self.model.to(self.device)
