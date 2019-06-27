#Author : Samantha Mahendran for RelaCy

from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os, logging, tempfile

class Model:

    def __init__(self, label, padding = False, common_words = 10000, maxlen = 100, embedding_dim =200 ):

        self.label = label
        self.padding = padding
        self.embedding_dim = embedding_dim
        self.common_words = common_words
        self.maxlen = maxlen


        #read dataset from external files
        train_data = self.read_from_file("sentence_train")
        train_labels = self.read_from_file("labels_train")
        test_data = self.read_from_file("sentence_test")
        test_labels = self.read_from_file("labels_test")

        # tokens = self.find_unique_tokens(train_data)

        #returns one-hot encoded train and test data and binarized train and test labels
        self.train, self.x_test, self.word_index = self.vectorizing_words(train_data, test_data)
        self.train_label = self.binarize_labels(train_labels)
        self.y_test = self.binarize_labels(test_labels)

        #divides train data into partial train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = self.create_validation_data(self.train, self.train_label)



    def read_from_file(self, file):
        """
        Function to read external files and insert the content to a list. It also removes whitespace
        characters like `\n` at the end of each line

        :param string: name of the input file.
        :return list:content of the file in list
        """

        if not os.path.isfile(file):
            raise FileNotFoundError("Not a valid file path")

        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        return content

    def find_unique_tokens(self, content_list):
        """
        Function takes a list as the input and tokenizes the samples via the `split` method.
        Assigns a unique index to each unique word and returns a dictionary of unique tokens

        :param list: name of the list.
        :return dictionary:dictionary with a unique index to each unique word
        """
        token_index = {}
        for content in content_list:
            for word in content.split():
                if word not in token_index:
                    token_index[word] = len(token_index) + 1
        return token_index

    def vectorizing_words(self, train_list,  test_list):

        """
        Function takes train and test lists as the input and creates a Keras tokenizer configured to only take
        into account the top given number most common words in the training data and builds the word index. Passing test data is optional.
        If test data is passed it will be tokenized using the tokenizer and output the ohe-hot-encoding
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
            test_sequences = tokenizer.texts_to_sequences(test_list)
            padded_train = pad_sequences(train_sequences, maxlen=self.maxlen)
            padded_test = pad_sequences(test_sequences, maxlen=self.maxlen)

        else:
            one_hot_train = tokenizer.texts_to_matrix(train_list, mode='binary')
            one_hot_test = tokenizer.texts_to_matrix(test_list, mode='binary')

        # To recover the word index that was computed
        word_index = tokenizer.word_index

        if self.padding:
            return padded_train, padded_test, word_index
        else:
            return one_hot_train, one_hot_test, word_index

    def binarize_labels(self, label_list):

        """
        Function takes a list as the input and binarizes labels in a one-vs-all fashion
        Then outputs the one-hot encoding of the input list

        :param list: list of text labels
        :return list:list of binarized labels
        """
        self.encoder = preprocessing.LabelBinarizer()
        self.encoder.fit(label_list)
        print(self.encoder.classes_)
        binary_label = self.encoder.transform(label_list)

        return binary_label

    def create_validation_data(self, train_data, train_label, num_data=1000):

        """
        Function takes the training data as the input and splits the data into training and validation.
        By default it takes first 1000 as the validation.
        :param list: train data
        :return lists:train samples, validation samples
        """

        x_val = train_data[:num_data]
        x_train = train_data[num_data:]

        y_val = train_label[:num_data]
        y_train = train_label[num_data:]

        return x_train, x_val, y_train, y_val

    def compute_evaluation_metrics(self, y_true, y_pred):
        precision, recall, fscore, support = score(y_true, y_pred)
        return precision, recall, fscore

    def compute_confusion_matrix(self, y_true, y_pred):
        matrix = confusion_matrix(y_true, y_pred)
        matrix.diagonal() / matrix.sum(axis=1)
        return matrix

    def plot_graphs(self, x_var, y_var, xlabel, ylabel, x_title, y_title, title):

        x_range = range(1, len(x_var) + 1)

        plt.plot(x_range, x_var, 'bo', label= x_title)
        plt.plot(x_range, y_var, 'b', label= y_title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        plt.show()