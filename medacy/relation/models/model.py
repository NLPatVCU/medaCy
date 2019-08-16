# Author : Samantha Mahendran for RelaCy

from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import os, logging, tempfile


def read_from_file(file):
    """
    Function to read external files and insert the content to a list. It also removes whitespace
    characters like `\n` at the end of each line

    :param file: name of the input file.
    :return list:content of the file in list
    """
    if not os.path.isfile(file):
        raise FileNotFoundError("Not a valid file path")

    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content


def create_validation_data(train_data, train_label, num_data=1000):
    """
    Function takes the training data as the input and splits the data into training and validation.
    By default it takes first 1000 as the validation.

    :param num_data: number of files split as validation data
    :param train_label: list of the labels of the training data
    :param train_data: list of the training data
    :return:train samples, validation samples
    """

    x_val = train_data[:num_data]
    x_train = train_data[num_data:]

    y_val = train_label[:num_data]
    y_train = train_label[num_data:]

    return x_train, x_val, y_train, y_val


def compute_evaluation_metrics(y_true, y_pred):
    precision, recall, f_score, support = score(y_true, y_pred)
    return precision, recall, f_score


def compute_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    matrix.diagonal() / matrix.sum(axis=1)
    return matrix


def plot_graphs(x_var, y_var, x_label, y_label, x_title, y_title, title):
    x_range = range(1, len(x_var) + 1)

    plt.plot(x_range, x_var, 'bo', label=x_title)
    plt.plot(x_range, y_var, 'b', label=y_title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.show()


class Model:

    def __init__(self, padding=False, segment=True, test=False, common_words=10000, maxlen=100):

        self.padding = padding
        self.segment = segment
        self.test = test
        self.common_words = common_words
        self.maxlen = maxlen

        # read dataset from external files
        train_data = read_from_file("cur/sentence_train")
        train_labels = read_from_file("cur/labels_train")
        if self.test:
            test_data = read_from_file("cur/sentence_test")
            test_labels = read_from_file("cur/labels_test")
        else:
            test_data = None
            test_labels = None

        self.train_label = train_labels
        # self.train_label = self.binarize_labels(train_labels, True)
        if self.test:
            self.train, self.x_test, self.word_index = self.vectorize_words(train_data, test_data)
            self.train_onehot, self.x_test_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
            self.y_test = self.binarize_labels(test_labels)
        else:
            self.train_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
            self.train, self.word_index = self.vectorize_words(train_data, test_data)

        # divides train data into partial train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = create_validation_data(self.train, self.train_label)
        self.x_train_onehot, self.x_val_onehot, self.y_train, self.y_val = create_validation_data(self.train_onehot,
                                                                                                  self.train_label)

        if segment:
            train_preceding = read_from_file("cur/preceding_seg")
            train_middle = read_from_file("cur/middle_seg")
            train_succeeding = read_from_file("cur/succeeding_seg")
            train_concept1 = read_from_file("cur/concept1_seg")
            train_concept2 = read_from_file("cur/concept2_seg")

            self.preceding, self.middle, self.succeeding, self.concept1, self.concept2, self.word_index = self.vectorize_segments(
                train_data, train_preceding, train_middle, train_succeeding, train_concept1, train_concept2)

    def one_hot_encoding(self, train_list, test_list=None):
        """
        Function takes a list as the input and tokenizes the samples via the `split` method.
        Assigns a unique index to each unique word and returns a dictionary of unique tokens.
        Then vectorize the train and test data passed and store the results in a matrix

        :param test_list:
        :param train_list:
        :param lists: train data, test data
        :return matrix:matrix with one-hot encoding
        """
        token_index = {}
        for content in train_list:
            for word in content.split():
                if word not in token_index:
                    token_index[word] = len(token_index) + 1

        # One_hot_encoding for train data
        one_hot_train = np.zeros((len(train_list), self.maxlen, max(token_index.values()) + 1))
        for i, sample in enumerate(train_list):
            for j, word in list(enumerate(sample.split()))[:self.maxlen]:
                index = token_index.get(word)
                one_hot_train[i, j, index] = 1.

        if self.test:
            # One_hot_encoding for test data
            one_hot_test = np.zeros((len(test_list), self.maxlen, max(token_index.values()) + 1))
            for i, sample in enumerate(test_list):
                for j, word in list(enumerate(sample.split()))[:self.maxlen]:
                    index = token_index.get(word)
                    one_hot_test[i, j, index] = 1.

            return one_hot_train, one_hot_test, token_index
        else:
            return one_hot_train, token_index

    def vectorize_words(self, train_list, test_list=None):

        """
        Function takes train and test lists as the input and creates a Keras tokenizer configured to only take
        into account the top given number most common words in the training data and builds the word index. Passing test data is optional.
        If test data is passed it will be tokenized using the tokenizer and output the one-hot-encoding
        Then directly outputs the one-hot encoding of the train, test and the word index

        :param list: name of the list
        :param integer: number of the common words (default = 10000)
        :return list:list with the outputs the one-hot encoding of the input list
        :return list:list with a unique index to each unique word
        """
        tokenizer = Tokenizer(self.common_words)
        # This builds the word index
        tokenizer.fit_on_texts(train_list)

        if self.padding:
            # Turns strings into lists of integer indices.
            train_sequences = tokenizer.texts_to_sequences(train_list)
            padded_train = pad_sequences(train_sequences, maxlen=self.maxlen)
            if self.test:
                test_sequences = tokenizer.texts_to_sequences(test_list)
                padded_test = pad_sequences(test_sequences, maxlen=self.maxlen)

        else:
            one_hot_train = tokenizer.texts_to_matrix(train_list, mode='binary')
            if self.test:
                one_hot_test = tokenizer.texts_to_matrix(test_list, mode='binary')

        # To recover the word index that was computed
        word_index = tokenizer.word_index

        if self.padding:
            if self.test:
                return padded_train, padded_test, word_index
            else:
                return padded_train, word_index
        else:
            if self.test:
                return one_hot_train, one_hot_test, word_index
            else:
                return one_hot_train, word_index

    def vectorize_segments(self, sentences, preceding, middle, succeeding, concept1, concept2):

        tokenizer = Tokenizer(self.common_words)
        # This builds the word index
        tokenizer.fit_on_texts(sentences)

        preceding_sequences = tokenizer.texts_to_sequences(preceding)
        padded_preceding = pad_sequences(preceding_sequences, maxlen=self.maxlen)

        middle_sequences = tokenizer.texts_to_sequences(middle)
        padded_middle = pad_sequences(middle_sequences, maxlen=self.maxlen)

        succeeding_sequences = tokenizer.texts_to_sequences(succeeding)
        padded_succeeding = pad_sequences(succeeding_sequences, maxlen=self.maxlen)

        concept1_sequences = tokenizer.texts_to_sequences(concept1)
        padded_concept1 = pad_sequences(concept1_sequences, maxlen=self.maxlen)

        concept2_sequences = tokenizer.texts_to_sequences(concept2)
        padded_concept2 = pad_sequences(concept2_sequences, maxlen=self.maxlen)

        # To recover the word index that was computed
        word_index = tokenizer.word_index

        return padded_preceding, padded_middle, padded_succeeding, padded_concept1, padded_concept2, word_index

    def binarize_labels(self, label_list, binarize=False):
        """
        Function takes a list as the input and binarizes labels in a one-vs-all fashion
        Then outputs the one-hot encoding of the input list

        :param label_list: list of text labels
        :return list:list of binarized labels
        """

        # if self.test or binarize:
        #     encoder = preprocessing.LabelBinarizer()
        # else:
        #     encoder = preprocessing.LabelEncoder()
        # encoder.fit(label_list)
        # binary_label = encoder.transform(label_list)

        if self.test or binarize:
            self.encoder = preprocessing.LabelBinarizer()
        else:
            self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(label_list)
        binary_label = self.encoder.transform(label_list)
        no_classes = len(self.encoder.classes_)

        true_binary_label = []

        if len(binary_label[0]) == 1:
            for label in binary_label:
                if label == 0:
                    true_binary_label.append([1, 0])
                else:
                    true_binary_label.append([0, 1])

        true_binary_label = np.array(true_binary_label)

        return true_binary_label