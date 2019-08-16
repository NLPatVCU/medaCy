from tabulate import tabulate
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from statistics import mean
from sklearn import preprocessing
from keras.models import *
from keras.layers import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging


def binarize_labels(train, binarize = True):
    """
    Function takes a list as the input and binarizes labels in a one-vs-all fashion
    Then outputs the one-hot encoding of the input list

    :param list: list of text labels
    :return list:list of binarized labels
    """

    if binarize:
        encoder = preprocessing.LabelBinarizer()
    else:
        encoder = preprocessing.LabelEncoder()
    encoder.fit(train)
    binary_train = encoder.transform(train)

    return binary_train, encoder.classes_


class Segment_CNN:

    def __init__(self, model, embedding=False):

        self.data_model = model
        self.embedding = embedding
        # self.labels = [str(i) for i in self.data_model.encoder.classes_]

    # define the model
    def define_model(self, no_classes, filters, filter_conv, filter_maxPool, activation, output_activation, drop_out):

        # channel 1
        inputs1 = Input(shape=(self.data_model.maxlen,))
        embedding1 = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                               weights=[self.embedding.embedding_matrix], trainable=False)(inputs1)
        # embedding1 = Embedding(self.data_model.common_words, self.embedding.embedding_dim )(inputs1)
        conv1 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding1)
        drop1 = Dropout(drop_out)(conv1)
        pool1 = MaxPooling1D(pool_size=filter_maxPool)(drop1)
        flat1 = Flatten()(pool1)

        # channel 2
        inputs2 = Input(shape=(self.data_model.maxlen,))
        embedding2 = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                               weights=[self.embedding.embedding_matrix], trainable=False)(inputs2)
        conv2 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding2)
        drop2 = Dropout(drop_out)(conv2)
        pool2 = MaxPooling1D(pool_size=filter_maxPool)(drop2)
        flat2 = Flatten()(pool2)

        # channel 3
        inputs3 = Input(shape=(self.data_model.maxlen,))
        embedding3 = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                               weights=[self.embedding.embedding_matrix], trainable=False)(inputs3)
        conv3 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding3)
        drop3 = Dropout(drop_out)(conv3)
        pool3 = MaxPooling1D(pool_size=filter_maxPool)(drop3)
        flat3 = Flatten()(pool3)

        # channel 4
        inputs4 = Input(shape=(self.data_model.maxlen,))
        embedding4 = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                               weights=[self.embedding.embedding_matrix], trainable=False)(inputs4)
        conv4 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding4)
        drop4 = Dropout(drop_out)(conv4)
        pool4 = MaxPooling1D(pool_size=filter_maxPool)(drop4)
        flat4 = Flatten()(pool4)

        # channel 5
        inputs5 = Input(shape=(self.data_model.maxlen,))
        embedding5 = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                               weights=[self.embedding.embedding_matrix], trainable=False)(inputs5)
        conv5 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding5)
        drop5 = Dropout(drop_out)(conv5)
        pool5 = MaxPooling1D(pool_size=filter_maxPool)(drop5)
        flat5 = Flatten()(pool5)

        # merge
        merged = concatenate([flat1, flat2, flat3, flat4, flat5])
        # interpretation
        dense1 = Dense(18, activation=activation)(merged)
        if no_classes == 2:
            outputs = Dense(1, activation=output_activation)(dense1)
        else:
            outputs = Dense(no_classes, activation=output_activation)(dense1)

        model = Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5], outputs=outputs)

        # compile
        if no_classes == 2:
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # summarize
        print(model.summary())
        return model

    def predict(self, model, x_test, y_test):

        pred = model.predict(x_test)
        y_pred_ind = np.argmax(pred, axis=1)
        y_true_ind = np.argmax(y_test, axis=1)
        y_pred = [self.data_model.encoder.classes_[i] for i in y_pred_ind]
        y_true = [self.data_model.encoder.classes_[i] for i in y_true_ind]
        test_loss, test_acc = model.evaluate(x_test, y_test)

        print("Accuracy :", test_acc)
        print("Loss : ", test_loss)

        return y_pred, y_true

    def evaluate_Model(self, y_pred, y_true):

        # print (classification_report(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=self.data_model.encoder.classes_))
        print(f1_score(y_true, y_pred, average='micro'))
        print(f1_score(y_true, y_pred, average='macro'))
        print(f1_score(y_true, y_pred, average='weighted'))

        matrix = confusion_matrix(y_true, y_pred)
        print(matrix)

    def cross_validate(self, Pre_data, Mid_data, Suc_data, C1_data, C2_data, Y_data, num_folds=5):

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        skf.get_n_splits( C1_data, Y_data)
        evaluation_statistics = {}
        fold = 1

        for train_index, test_index in skf.split([Pre_data, Mid_data, Suc_data, C1_data, C2_data], Y_data):

            binary_Y = self.data_model.binarize_labels(Y_data, True)
            # encoder = preprocessing.LabelBinarizer()
            # encoder.fit(Y_data)
            # binary_Y = encoder.transform(Y_data)

            pre_train, pre_test = Pre_data[train_index], Pre_data[test_index]
            mid_train, mid_test = Mid_data[train_index], Mid_data[test_index]
            suc_train, suc_test = Suc_data[train_index], Suc_data[test_index]
            c1_train, c1_test = C1_data[train_index], C1_data[test_index]
            c2_train, c2_test = C2_data[train_index], C2_data[test_index]
            y_train, y_test = binary_Y[train_index], binary_Y[test_index]

            # binary_train, binary_test, encoder_classes = binarize_labels(y_train, y_test)
            labels = [str(i) for i in self.data_model.encoder.classes_]

            fold_statistics = {}

            # num_val_samples = len(C1_data) // num_folds
            # evaluation_statistics = {}

            # for i in range(num_folds):
            #     fold_statistics = {}
            #
            #     print('processing fold #', i)
            #     # Prepare the validation data: data from partition # k
            #     pre_test = Pre_data[i * num_val_samples: (i + 1) * num_val_samples]
            #     mid_test = Mid_data[i * num_val_samples: (i + 1) * num_val_samples]
            #     suc_test = Suc_data[i * num_val_samples: (i + 1) * num_val_samples]
            #     c1_test = C1_data[i * num_val_samples: (i + 1) * num_val_samples]
            #     c2_test = C2_data[i * num_val_samples: (i + 1) * num_val_samples]
            #     y_test = Y_data[i * num_val_samples: (i + 1) * num_val_samples]
            #
            #     # Prepare the training data: data from all other partitions
            #     pre_train = np.concatenate([Pre_data[:i * num_val_samples],Pre_data[(i + 1) * num_val_samples:]], axis=0)
            #     mid_train = np.concatenate([Mid_data[:i * num_val_samples], Mid_data[(i + 1) * num_val_samples:]], axis=0)
            #     suc_train = np.concatenate([Suc_data[:i * num_val_samples], Suc_data[(i + 1) * num_val_samples:]], axis=0)
            #     c1_train = np.concatenate([C1_data[:i * num_val_samples], C1_data[(i + 1) * num_val_samples:]], axis=0)
            #     c2_train = np.concatenate([C2_data[:i * num_val_samples], C2_data[(i + 1) * num_val_samples:]], axis=0)
            #     y_train = np.concatenate([Y_data[:i * num_val_samples],Y_data[(i + 1) * num_val_samples:]], axis=0)



            model = self.define_model(len(self.data_model.encoder.classes_), 32, 1, 5, 'relu', 'sigmoid', 0.5)
            model.fit([pre_train, mid_train, suc_train, c1_train, c2_train], y_train, epochs=20, batch_size=512)
            y_pred, y_true = self.predict(model, [pre_test, mid_test, suc_test, c1_test, c2_test], y_test)

            # Write the metrics for this fold.
            for label in labels:
                fold_statistics[label] = {}
                f1 = f1_score(y_true, y_pred, average='micro', labels=[label])
                precision = precision_score(y_true, y_pred, average='micro', labels=[label])
                recall = recall_score(y_true, y_pred, average='micro', labels=[label])
                fold_statistics[label]['precision'] = precision
                fold_statistics[label]['recall'] = recall
                fold_statistics[label]['f1'] = f1

            # add averages
            fold_statistics['system'] = {}
            f1 = f1_score(y_true, y_pred, average='micro')
            precision = precision_score(y_true, y_pred, average='micro')
            recall = recall_score(y_true, y_pred, average='micro')
            fold_statistics['system']['precision'] = precision
            fold_statistics['system']['recall'] = recall
            fold_statistics['system']['f1'] = f1

            table_data = [[label,
                           format(fold_statistics[label]['precision'], ".3f"),
                           format(fold_statistics[label]['recall'], ".3f"),
                           format(fold_statistics[label]['f1'], ".3f")]
                          for label in labels + ['system']]

            print(tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1'],
                           tablefmt='orgtbl'))

            evaluation_statistics[fold] = fold_statistics
            fold +=1

        statistics_all_folds = {}

        for label in labels + ['system']:
            statistics_all_folds[label] = {}
            statistics_all_folds[label]['precision_average'] = mean(
                [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])
            statistics_all_folds[label]['precision_max'] = max(
                [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])
            statistics_all_folds[label]['precision_min'] = min(
                [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])

            statistics_all_folds[label]['recall_average'] = mean(
                [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])
            statistics_all_folds[label]['recall_max'] = max(
                [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])
            statistics_all_folds[label]['recall_min'] = min(
                [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])

            statistics_all_folds[label]['f1_average'] = mean(
                [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])
            statistics_all_folds[label]['f1_max'] = max(
                [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])
            statistics_all_folds[label]['f1_min'] = min(
                [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])

        table_data = [[label,
                       format(statistics_all_folds[label]['precision_average'], ".3f"),
                       format(statistics_all_folds[label]['recall_average'], ".3f"),
                       format(statistics_all_folds[label]['f1_average'], ".3f"),
                       format(statistics_all_folds[label]['f1_min'], ".3f"),
                       format(statistics_all_folds[label]['f1_max'], ".3f")]
                      for label in labels + ['system']]

        print("\n" + tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
                              tablefmt='orgtbl'))
