#Author : Samantha Mahendran for RelaCy

from tabulate import tabulate
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.models import *
from keras import layers
from keras.layers import *
from statistics import mean
import numpy as np
import logging

class CNN:

    def __init__(self, model, embedding = False):
        self.data_model = model
        self.embedding = embedding
        # self.labels = [str(i) for i in self.data_model.encoder.classes_]

    def build_Model(self, hidden_units = 64, filter_conv = 1, filter_maxPool = 5, hidden_activation = 'relu',
                    output_activation = 'sigmoid', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy']):
        model = Sequential()
        model.add(layers.Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                   input_length=self.data_model.maxlen))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation= hidden_activation))
        model.add(layers.MaxPooling1D(filter_maxPool))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation= hidden_activation))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(len(self.data_model.encoder.classes_), activation=output_activation))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


    def define_Embedding_Model(self, no_classes, hidden_units = 64, filter_conv = 1, filter_maxPool = 5, hidden_activation ='relu',
                               output_activation = 'sigmoid', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy']):
        model = Sequential()
        model.add(layers.Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                   input_length=self.data_model.maxlen))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation= hidden_activation))
        model.add(layers.Dropout(0.5))
        model.add(layers.MaxPooling1D(filter_maxPool))
        # model.add(layers.Conv1D(hidden_units, filter_conv, activation=hidden_activation))
        # model.add(layers.Dropout(0.5))
        # model.add(layers.MaxPooling1D(filter_maxPool))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation= hidden_activation))
        model.add(layers.GlobalMaxPooling1D())
        # model.add(layers.Flatten())
        if self.embedding:
            model.layers[0].set_weights([self.embedding.embedding_matrix])
            model.layers[0].trainable = False

        if no_classes == 2:
            model.add(layers.Dense(1, activation=output_activation))
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        else:
            model.add(layers.Dense(len(self.data_model.encoder.classes_), activation=output_activation))
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        print(model.summary())

        return model

    def fit_Model(self, model, x_train, y_train, no_epochs = 20, batch_Size = 512, validation = None):

        history = model.fit(x_train, y_train, epochs= no_epochs,
                            batch_size= batch_Size, validation_data=validation)
        print("epochs: ", no_epochs)
        loss = history.history['loss']
        acc = history.history['acc']
        if validation is not None:
            val_loss = history.history['val_loss']
            val_acc = history.history['val_acc']
            max_epoch = val_acc.index(max(val_acc))+1
            self.data_model.plot_graphs(loss, val_loss, 'Epochs','Loss','Training loss', 'Validation loss','Training and validation loss' )
            self.data_model.plot_graphs(acc, val_acc, 'Epochs', 'Acc', 'Training acc', 'Validation acc','Training and validation acc')
            return model, loss, val_loss, acc, val_acc, max_epoch

        return model, loss, acc

    def predict(self, model, x_test, y_test ):

        pred = model.predict(x_test)
        y_pred_ind = np.argmax(pred, axis=1)
        y_true_ind = np.argmax(y_test, axis=1)
        y_pred = [self.data_model.encoder.classes_[i] for i in y_pred_ind]
        y_true = [self.data_model.encoder.classes_[i] for i in y_true_ind]
        test_loss, test_acc = model.evaluate(x_test, y_test)

        print ("Accuracy :", test_acc)
        print ("Loss : ", test_loss)

        return y_pred, y_true

    def evaluate_Model(self, y_pred, y_true ):

        # print (classification_report(y_true, y_pred))
        print (classification_report(y_true, y_pred, target_names=self.data_model.encoder.classes_))
        print(f1_score(y_true, y_pred, average='micro'))
        print(f1_score(y_true, y_pred, average='macro'))
        print(f1_score(y_true, y_pred, average='weighted') )

        matrix = confusion_matrix(y_true, y_pred)
        print (matrix)

    def cross_validate(self, X_data, Y_data, num_folds = 5):

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        assert X_data is not None and Y_data is not None, \
            "Must have features and labels extracted for cross validation"

        skf = StratifiedKFold(n_splits = num_folds, shuffle=True)
        skf.get_n_splits(X_data, Y_data)

        evaluation_statistics = {}
        fold = 1
        for train_index, test_index in skf.split(X_data, Y_data):
            binary_Y = self.data_model.binarize_labels(Y_data, True)
            x_train, x_test = X_data[train_index], X_data[test_index]
            y_train, y_test = binary_Y[train_index], binary_Y[test_index]
            print("Training Fold %i", fold)

            labels = [str(i) for i in self.data_model.encoder.classes_]
            fold_statistics = {}

        # num_val_samples = len(X_data) // num_folds
        # evaluation_statistics = {}
        # for i in range(num_folds):
        #     fold_statistics = {}
        #
        #     print('processing fold #', i)
        #     # Prepare the validation data: data from partition # k
        #     x_test = X_data[i * num_val_samples: (i + 1) * num_val_samples]
        #     y_test = Y_data[i * num_val_samples: (i + 1) * num_val_samples]
        #
        #     # Prepare the training data: data from all other partitions
        #     x_train = np.concatenate(
        #         [X_data[:i * num_val_samples],
        #          X_data[(i + 1) * num_val_samples:]],
        #         axis=0)
        #     y_train = np.concatenate(
        #         [Y_data[:i * num_val_samples],
        #          Y_data[(i + 1) * num_val_samples:]],
        #         axis=0)

            model_CNN = self.define_Embedding_Model(len(self.data_model.encoder.classes_))
            model, loss, acc = self.fit_Model (model_CNN, x_train, y_train)
            y_pred, y_true = self.predict(model,x_test, y_test)

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
            fold += 1

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

    # def cross_validate(self, X_data, Y_data, num_folds=5):
    #
    #     if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")
    #
    #     assert X_data is not None and Y_data is not None, \
    #         "Must have features and labels extracted for cross validation"
    #
    #     num_val_samples = len(X_data) // num_folds
    #     evaluation_statistics = {}
    #     for i in range(num_folds):
    #         fold_statistics = {}
    #
    #         print('processing fold #', i)
    #         # Prepare the validation data: data from partition # k
    #         x_test = X_data[i * num_val_samples: (i + 1) * num_val_samples]
    #         y_test = Y_data[i * num_val_samples: (i + 1) * num_val_samples]
    #
    #         # Prepare the training data: data from all other partitions
    #         x_train = np.concatenate(
    #             [X_data[:i * num_val_samples],
    #              X_data[(i + 1) * num_val_samples:]],
    #             axis=0)
    #         y_train = np.concatenate(
    #             [Y_data[:i * num_val_samples],
    #              Y_data[(i + 1) * num_val_samples:]],
    #             axis=0)
    #
    #         model_CNN = self.define_Embedding_Model()
    #         model, loss, acc = self.fit_Model(model_CNN, x_train, y_train)
    #         y_pred, y_true = self.predict(model, x_test, y_test)
    #
    #         # Write the metrics for this fold.
    #         for label in self.labels:
    #             fold_statistics[label] = {}
    #             f1 = f1_score(y_true, y_pred, average='micro', labels=[label])
    #             precision = precision_score(y_true, y_pred, average='macro', labels=[label])
    #             recall = recall_score(y_true, y_pred, average='micro', labels=[label])
    #             fold_statistics[label]['precision'] = precision
    #             fold_statistics[label]['recall'] = recall
    #             fold_statistics[label]['f1'] = f1
    #
    #         # add averages
    #         fold_statistics['system'] = {}
    #         f1 = f1_score(y_true, y_pred, average='micro')
    #         precision = precision_score(y_true, y_pred, average='macro')
    #         recall = recall_score(y_true, y_pred, average='micro')
    #         fold_statistics['system']['precision'] = precision
    #         fold_statistics['system']['recall'] = recall
    #         fold_statistics['system']['f1'] = f1
    #
    #         table_data = [[label,
    #                        format(fold_statistics[label]['precision'], ".3f"),
    #                        format(fold_statistics[label]['recall'], ".3f"),
    #                        format(fold_statistics[label]['f1'], ".3f")]
    #                       for label in self.labels + ['system']]
    #
    #         print(tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1'],
    #                        tablefmt='orgtbl'))
    #
    #         evaluation_statistics[i + 1] = fold_statistics
    #
    #     statistics_all_folds = {}
    #
    #     for label in self.labels + ['system']:
    #         statistics_all_folds[label] = {}
    #         statistics_all_folds[label]['precision_average'] = mean(
    #             [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])
    #         statistics_all_folds[label]['precision_max'] = max(
    #             [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])
    #         statistics_all_folds[label]['precision_min'] = min(
    #             [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])
    #
    #         statistics_all_folds[label]['recall_average'] = mean(
    #             [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])
    #         statistics_all_folds[label]['recall_max'] = max(
    #             [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])
    #         statistics_all_folds[label]['recall_min'] = min(
    #             [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])
    #
    #         statistics_all_folds[label]['f1_average'] = mean(
    #             [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])
    #         statistics_all_folds[label]['f1_max'] = max(
    #             [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])
    #         statistics_all_folds[label]['f1_min'] = min(
    #             [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])
    #
    #     table_data = [[label,
    #                    format(statistics_all_folds[label]['precision_average'], ".3f"),
    #                    format(statistics_all_folds[label]['recall_average'], ".3f"),
    #                    format(statistics_all_folds[label]['f1_average'], ".3f"),
    #                    format(statistics_all_folds[label]['f1_min'], ".3f"),
    #                    format(statistics_all_folds[label]['f1_max'], ".3f")]
    #                   for label in self.labels + ['system']]
    #
    #     print("\n" + tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
    #                           tablefmt='orgtbl'))