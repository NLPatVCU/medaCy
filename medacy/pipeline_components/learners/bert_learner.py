import logging
import os
import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import AdamW, BertTokenizer, BertForTokenClassification

from medacy.nn import Vectorizer, SequencesDataset

class BertLearner:
    """Learner for running predictions and fine tuning BERT models.
    """

    def __init__(self, cuda_device=-1, pretrained_model='bert-large-cased', batch_size=32):
        """
        Initialize BertLearner.

        :param cuda_device: CUDA device to use when running on a GPU. Use -1 for CPU.
        :param pretrained_model: Name of model to use for the Transformers pretrained_model. This
            is different from the medaCy fine-tuned model you may have created.
        """
        torch.manual_seed(1) # Seed PyTorch for consistency

        # Create torch.device that all tensors will be using
        device_string = 'cuda:%d' % cuda_device if cuda_device >= 0 else 'cpu'
        self.device = torch.device(device_string)

        self.model = None # Transformers/PyTorch model
        self.tokenizer = None # Transformers tokenizer
        self.vectorizer = Vectorizer(self.device) # medaCy vectorizer
        self.pretrained_model = pretrained_model # Name of Transformers pretrained model
        self.batch_size = batch_size

    def encode_sequences(self, sequences, labels=[]):
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
            for sequence in sequences:
                labels.append(['O'] * len(sequence))

        # Use the tokenizer to encode every sequence and add in special tokens like [CLS] and [SEP]
        # Then add them to encoded_sequences
        for sequence in sequences:
            encoded_sequence = []
            for token in sequence:
                encoded_sequence.append(self.tokenizer.encode(token, add_special_tokens=False))
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
                if not ids:
                    ids = [self.tokenizer.unk_token_id]
                split_sequence.append(ids[0])
                split_labels.append(label)

                for token_id in ids[1:]:
                    # Add proper token id and label to lists
                    split_sequence.append(token_id)
                    split_labels.append('X') # 'X' marks a token we should ignore

            # Add the final token [SEP] with proper label/mapping
            split_sequence.append(sequence[-1])
            split_labels.append('O')

            # Append final forms of lists
            split_sequences.append(split_sequence)
            split_sequence_labels.append(self.vectorizer.vectorize_tags(split_labels))

        # Return the two lists that were created
        return split_sequences, split_sequence_labels

    def decode_labels(self, sequence_labels, mappings):
        """
        Decode list of label indices using mappings generated during self.encode_sequences()

        :param sequence_labels: List of list of label indices. Index corresponds to index of label
            in self.vectorizer
        :param mappings: List of list of mappings. All mappings indicate whether we should include
            it in classification (1) or not (0)
        """
        decoded_labels = []
        null_label = self.vectorizer.tag_to_index['X']

        for labels, mapping in zip(sequence_labels, mappings):
            remapped_labels = []

            for label, map_value in zip(labels[1:-1], mapping[1:-1]): # Ignore special tokens
                # Only include label if map value is 1
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
        self.vectorizer.create_tag_dictionary(y_data)
        self.vectorizer.add_tag('X')

        # Load pretrain BERT model, unfreeze layers, move it to GPU device, and create its tokenizer
        self.model = BertForTokenClassification.from_pretrained(
            self.pretrained_model,
            num_labels=len(self.vectorizer.tag_to_index) - 1 # Don't include 'X'
        )
        self.model.train()
        self.model.to(device=self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)

        # Use Adam optimizer with weight decay options as shown in the link below:
        # https://github.com/huggingface/transformers/blob/master/examples/run_ner.py
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

        # Encode sequences and labels
        sequences, labels = self.encode_sequences(x_data, y_data)
        dataset = SequencesDataset(
            self.device,
            sequences,
            self.vectorizer.tag_to_index['X'],
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
        
        # Only do 3 epochs as suggested in BERT paper
        for epoch in range(3):
            logging.info('Epoch %d' % epoch)
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

            logging.info('Loss: %f' % (training_loss / batches))

    def predict(self, x_data):
        """Use model to make predictions over a given dataset.

        :param sequences: Sequences to predict labels for.
        :return: List of list of predicted labels.
        """
        self.model.eval()
        encoded_sequences, mappings = self.encode_sequences(x_data, labels=[])
        encoded_tag_indices = []

        dataset = SequencesDataset(
            device=self.device,
            sequences=encoded_sequences,
            mask_label=self.vectorizer.tag_to_index['X'],
            sequence_labels=mappings,
            o_label=self.vectorizer.tag_to_index['O']
        )
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=dataset.collate
        )

        for batch in dataloader:
            sequences, attention_masks, _ = batch
            scores = self.model(sequences, attention_mask=attention_masks)[0]
            tag_indices = torch.max(scores, 2)[1].tolist()
            encoded_tag_indices.extend(tag_indices)

        predictions = self.decode_labels(encoded_tag_indices, mappings)
        return predictions

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        vectorizer_values = self.vectorizer.get_values()
        torch.save(vectorizer_values, path + '/vectorizer.pt')

    def load(self, path):
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        vectorizer_values = torch.load(path + '/vectorizer.pt')
        self.vectorizer = Vectorizer(device=self.device)
        self.vectorizer.load_values(vectorizer_values)
        self.model = BertForTokenClassification.from_pretrained(
            path,
            num_labels=len(self.vectorizer.tag_to_index) - 1 # Ignore 'X'
        )
