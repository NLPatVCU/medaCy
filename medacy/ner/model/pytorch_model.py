from os import makedirs
from os.path import join, isdir
import logging
from tabulate import tabulate
from sklearn_crfsuite import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
from spacy.gold import biluo_tags_from_offsets
from medacy.data import Dataset
from medacy.tools import Annotations, BiluoTokenizer

import sys

# Constants
SEGMENT_SIZE = 100

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class PytorchModel:
    # Properties
    model = None
    labels = []
    word_to_index = {}
    tag_to_index = {}

    def __init__(self, bidirectional=False):
        self.bidirectional = bidirectional

        torch.manual_seed(1)

    def vectorize(self, sequence, to_index):
        indexes = [to_index[w] for w in sequence]
        return torch.tensor(indexes, dtype=torch.long)

    def devectorize(self, vectors, to_index):
        to_tag = {y:x for x, y in to_index.items()}
        tags = []

        for vector in vectors:
            max_value = max(vector)
            index = list(vector).index(max_value)
            tags.append(to_tag[index])
            
        return tags

    def get_segments(self, source):
        segments = []

        for i in range(0, len(source), SEGMENT_SIZE):
            segments.append(source[i:i + SEGMENT_SIZE])

        return segments

    def get_training_data(self, dataset):
        labels = dataset.get_labels()
        labels.add('O')
        training_data = dataset.get_training_data('pytorch')

        word_to_index = {}
        for sent, tags in training_data:
            for word in sent:
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index)

        tag_to_index = {}

        for label in labels:
            tag_to_index[label] = len(tag_to_index)

        preprocessed_data = []

        for document, tags in training_data:
            sentences = self.get_segments(document)
            tag_segments = self.get_segments(tags)

            for i in range(len(sentences)):
                preprocessed_data.append((sentences[i], tag_segments[i]))

        self.labels = labels
        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index

        return preprocessed_data

    def fit(self, dataset, epochs=30):
        if not isinstance(dataset, Dataset):
            raise TypeError("Must pass in an instance of Dataset containing your training files")

        training_data = self.get_training_data(dataset)

        word_to_index = self.word_to_index
        tag_to_index = self.tag_to_index

        # These will usually be more like 32 or 64 dimensional.
        # We will keep them small, so we can see how the weights change as we train.
        EMBEDDING_DIM = 64
        HIDDEN_DIM = 64

        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(tag_to_index), self.bidirectional)
        loss_weights = torch.ones(len(tag_to_index))
        loss_weights[0] = 0.01 # Reduce weighting towards 'O' label
        loss_function = nn.NLLLoss(loss_weights)
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(epochs):
            print('Epoch %d' % epoch)
            losses = []
            i = 0
            for sentence, tags in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                optimizer.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = self.vectorize(sentence, word_to_index)
                targets = self.vectorize(tags, tag_to_index)

                # Step 3. Run our forward pass.
                tag_scores = model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(loss)
                i += 1
                losses.append(loss)

            average_loss = sum(losses) / len(losses)
            print('AVG LOSS: %f' % average_loss)

        self.model = model

    def predict(self, dataset, prediction_directory=None):
        """
        Generates predictions over a string or a medaCy dataset

        :param dataset: a string or medaCy Dataset to predict
        :param prediction_directory: the directory to write predictions if doing bulk prediction
                                     (default: */prediction* sub-directory of Dataset)
        """
        if not isinstance(dataset, (Dataset, str)):
            raise TypeError("Must pass in an instance of Dataset")
        if self.model is None:
            raise ValueError("Must fit or load a pickled model before predicting")

        model = self.model

        if isinstance(dataset, Dataset):
            if prediction_directory is None:
                prediction_directory = str(dataset.data_directory) + "/predictions/"

            if isdir(prediction_directory):
                logging.warning("Overwriting existing predictions")
            else:
                makedirs(prediction_directory)

            predictions = []

            for data_file in dataset.get_data_files():
                logging.info("Predicting file: %s", data_file.file_name)

                with open(data_file.get_text_path(), 'r') as source_text_file:
                    text = source_text_file.read()

                biluo_tokenizer = BiluoTokenizer(text)
                tokens = biluo_tokenizer.get_tokens()
                segmented_tokens = self.get_segments(tokens)

                with torch.no_grad():
                    devectorized_tags = []
                    for sequence in segmented_tokens:
                        inputs = self.vectorize(sequence, self.word_to_index)
                        tag_scores = model(inputs)
                        devectorized_tags.extend(self.devectorize(tag_scores, self.tag_to_index))
                    # entities = biluo_tokenizer.get_entities(devectorized_tags)
                    predictions.append(devectorized_tags)

            return predictions

                # for ent in doc.ents:
                #     predictions.append((ent.label_, ent.start_char, ent.end_char, ent.text))

                # annotations = construct_annotations_from_tuples(text, predictions)

                # prediction_filename = join(prediction_directory, data_file.file_name + ".ann")
                # logging.debug("Writing to: %s", prediction_filename)
                # annotations.to_ann(write_location=prediction_filename)

        if isinstance(dataset, str):
            doc = nlp(dataset)

            entities = []

            for ent in doc.ents:
                entities.append((ent.start_char, ent.end_char, ent.label_))

            return entities

    def get_scores(self, predictions, actuals, labels):
        scores = {}

        scores['recall'] = metrics.flat_recall_score(
            actuals,
            predictions,
            average='weighted',
            labels=labels
        )

        scores['precision'] = metrics.flat_precision_score(
            actuals,
            predictions,
            average='weighted',
            labels=labels
        )

        scores['f1'] = metrics.flat_f1_score(
            actuals,
            predictions,
            average='weighted',
            labels=labels
        )

        return scores


    def evaluate(self, predictions, actuals):
        labels = list(self.labels)
        labels.remove('O')
        scores = {}
        
        # See what the scores are after training
        for label in self.labels:
            scores[label] = self.get_scores(predictions, actuals, [label])

        scores['system'] = self.get_scores(predictions, actuals, labels)

        return scores

    def cross_validate(self, dataset):
        model = self.model
        labels = list(self.labels)
        labels.remove('O')
        print(labels)

        print('Getting training data...')
        training_data = dataset.get_training_data('pytorch')
        print('Finished fixing annotations.')

        actuals = [tags for (_, tags) in training_data]
        predictions = self.predict(dataset)

        scores = self.evaluate(predictions, actuals)
        
        table_data = []

        for label in labels + ['system']:
            entry = [
                label,
                format(scores[label]['precision'], ".3f"),
                format(scores[label]['recall'], ".3f"),
                format(scores[label]['f1'], ".3f")
            ]

            table_data.append(entry)

        print()
        print(tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1'],
                                tablefmt='orgtbl'))

    def save(self, path='nameless-model.pt'):
        torch.save({
            'model': self.model,
            'word_to_index': self.word_to_index,
            'tag_to_index': self.tag_to_index,
            'labels': self.labels
        }, path)

    def load(self, path='nameless-model.pt'):
        saved_data = torch.load(path)
        
        self.word_to_index = saved_data['word_to_index']
        self.tag_to_index = saved_data['tag_to_index']
        self.labels = saved_data['labels']

        model = saved_data['model']
        model.eval()
        self.model = model
