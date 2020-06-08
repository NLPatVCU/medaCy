"""
BiLSTM+CRF PyTorch network and model.
"""
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from medacy.pipeline_components.learners.nn.bilstm_crf import BiLstmCrf
from medacy.pipeline_components.learners.nn.vectorizer import Vectorizer


class BiLstmCrfLearner:
    """
    BiLSTM-CRF model class for using the network. Currently handles all vectorization as well.

    :ivar device: PyTorch device to use.
    :ivar model: Instance of BiLstmCrfNetwork to use.
    :ivar word_embeddings_file: File to load word embeddings from.
    :ivar word_vectors: Gensim word vectors object for use in configuring word embeddings.
    """

    def __init__(self, word_embeddings, cuda_device):
        """Init BiLstmCrfLearner object.

        :param word_embeddings: Path to word embeddings file to use.
        :param cuda_device: Index of cuda device to use. Use -1 to use CPU.
        """
        torch.manual_seed(1)
        device_string = 'cuda:%d' % cuda_device if cuda_device >= 0 else 'cpu'
        self.device = torch.device(device_string)

        self.vectorizer = Vectorizer(self.device)

        if word_embeddings is None:
            raise ValueError('BiLSTM-CRF requires word embeddings.')
        else:
            self.word_embeddings_file = word_embeddings
            self.word_vectors = None

        # Other instance attributes
        self.model = None
        self.learning_rate = 0.01

        # TODO: Implement cleaner way to handle this
        if word_embeddings.endswith('test_word_embeddings.txt'):
            self.epochs = 2
            self.crf_delay = 1
        else:
            self.epochs = 40
            self.crf_delay = 20

    def fit(self, x_data, y_data):
        """Fully train model based on x and y data. self.model is set to trained model.

        :param x_data: List of list of tokens.
        :param y_data: List of list of correct labels for the tokens.
        """
        if self.vectorizer.word_vectors is None:
            self.vectorizer.load_word_embeddings(self.word_embeddings_file)

        data = self.vectorizer.vectorize_dataset(x_data, y_data)

        # Create network
        model = BiLstmCrf(
            self.vectorizer.word_vectors,
            len(data[0][0][0][2:]),
            len(self.vectorizer.tag_to_index),
            self.device
        )

        # Move to GPU if possible
        if self.device.type != 'cpu':
            logging.info('CUDA available. Moving model to GPU.')
            model = model.to(self.device)

        # Setup optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_function = nn.NLLLoss()

        logging.info('Training BiLSTM-CRF...')

        # Training loop
        for i in range(1, self.epochs + 1):
            random.shuffle(data)
            epoch_losses = []

            for sentence, sentence_tags in data:
                if i <= self.crf_delay:
                    emissions = model(sentence)
                    predictions = F.log_softmax(emissions, dim=1)
                    loss = loss_function(predictions, sentence_tags)
                else:
                    emissions = model(sentence).unsqueeze(1)
                    sentence_tags = sentence_tags.unsqueeze(1)
                    loss = -model.crf(emissions, sentence_tags)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)

            average_loss = sum(epoch_losses) / len(epoch_losses)
            logging.info('Epoch %d average loss: %f', i, average_loss)

        self.model = model

    def predict(self, sequences):
        """Use model to make predictions over a given dataset.

        :param sequences: Sequences to predict labels for.
        :return: List of list of predicted labels.
        """
        if not self.vectorizer.word_vectors:
            raise RuntimeError('Loading word embeddings is required.')

        with torch.no_grad():
            predictions = []
            for sequence in sequences:
                vectorized_tokens = self.vectorizer.vectorize_tokens(sequence)
                emissions = self.model(vectorized_tokens).unsqueeze(1)
                tag_indices = self.model.crf.decode(emissions)
                predictions.append(self.vectorizer.devectorize_tag(tag_indices[0]))

        return predictions

    def save(self, path):
        """Save model and other required values.

        :param path: Path to save model to.
        """
        vectorizer_values = self.vectorizer.get_values()

        properties = {
            'model': self.model,
            'vectorizer_values': vectorizer_values,
        }

        if not path.endswith('.pt'):
            path += '.pt'

        torch.save(properties, path)

    def load(self, path):
        """Load model and other required values from given path.

        :param path: Path of saved model.
        """
        saved_data = torch.load(path, map_location=self.device)

        self.vectorizer.load_values(saved_data['vectorizer_values'])

        model = saved_data['model']
        model.device = self.device
        model.eval()

        self.model = model.to(self.device)
        self.model.device = self.device
        self.vectorizer.load_word_embeddings(self.word_embeddings_file)
