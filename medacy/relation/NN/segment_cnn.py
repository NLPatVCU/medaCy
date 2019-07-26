from tabulate import tabulate
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import *
from keras.layers import *
from statistics import mean
import numpy as np
import logging

class Segment_CNN:

    def __init__(self, model, embedding = False):
        self.data_model = model
        self.embedding = embedding

    # define the model
    def define_model(self, filters, filter_conv, filter_maxPool, activation, output_activation, drop_out):

        # channel 1
        inputs1 = Input(shape=(self.data_model.maxlen,))
        embedding1 = Embedding(self.data_model.common_words, self.embedding.embedding_dim, weights=[self.embedding.embedding_matrix], trainable=False)(inputs1)
        # embedding1 = Embedding(self.data_model.common_words, self.embedding.embedding_dim )(inputs1)
        conv1 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding1)
        drop1 = Dropout(drop_out)(conv1)
        pool1 = MaxPooling1D(pool_size= filter_maxPool)(drop1)
        flat1 = Flatten()(pool1)

        # channel 2
        inputs2 = Input(shape=(self.data_model.maxlen,))
        embedding2 = Embedding(self.data_model.common_words, self.embedding.embedding_dim, weights=[self.embedding.embedding_matrix], trainable=False)(inputs2)
        conv2 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding2)
        drop2 = Dropout(drop_out)(conv2)
        pool2 = MaxPooling1D(pool_size= filter_maxPool)(drop2)
        flat2 = Flatten()(pool2)

        # channel 3
        inputs3 = Input(shape=(self.data_model.maxlen,))
        embedding3 = Embedding(self.data_model.common_words, self.embedding.embedding_dim, weights=[self.embedding.embedding_matrix], trainable=False)(inputs3)
        conv3 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding3)
        drop3 = Dropout(drop_out)(conv3)
        pool3 = MaxPooling1D(pool_size= filter_maxPool)(drop3)
        flat3 = Flatten()(pool3)

        # channel 4
        inputs4 = Input(shape=(self.data_model.maxlen,))
        embedding4 = Embedding(self.data_model.common_words, self.embedding.embedding_dim, weights=[self.embedding.embedding_matrix], trainable=False)(inputs4)
        conv4 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding4)
        drop4 = Dropout(drop_out)(conv4)
        pool4 = MaxPooling1D(pool_size= filter_maxPool)(drop4)
        flat4 = Flatten()(pool4)

        # channel 5
        inputs5 = Input(shape=(self.data_model.maxlen,))
        embedding5 = Embedding(self.data_model.common_words, self.embedding.embedding_dim, weights=[self.embedding.embedding_matrix], trainable=False)(inputs5)
        conv5 = Conv1D(filters=filters, kernel_size=filter_conv, activation=activation)(embedding5)
        drop5 = Dropout(drop_out)(conv5)
        pool5 = MaxPooling1D(pool_size= filter_maxPool)(drop5)
        flat5 = Flatten()(pool5)

        # merge
        merged = concatenate([flat1, flat2, flat3, flat4, flat5])

        # interpretation
        dense1 = Dense(18, activation= activation)(merged)
        outputs = Dense(len(self.data_model.label), activation=output_activation)(dense1)

        model = Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5], outputs=outputs)

        # compile
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # summarize
        print(model.summary())
        return model

    def predict(self, model, x_test, y_test ):

        pred = model.predict(x_test)
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        test_loss, test_acc = model.evaluate(x_test, y_test)

        print ("Accuracy :", test_acc)
        print ("Loss : ", test_loss)

        return y_pred, y_true

    def evaluate_Model(self, y_pred, y_true ):

        print (classification_report(y_true, y_pred))
        # print (classification_report(y_true, y_pred, target_names=self.data_model.label))
        print(f1_score(y_true, y_pred, average='micro'))
        print(f1_score(y_true, y_pred, average='macro'))
        print(f1_score(y_true, y_pred, average='weighted') )

        matrix = confusion_matrix(y_true, y_pred)
        print (matrix)

    def cross_validate(self, Pre_data, Mid_data, Suc_data, C1_data, C2_data, Y_data, num_folds = 5):

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        num_val_samples = len(C1_data) // num_folds
        evaluation_statistics = {}

        for i in range(num_folds):
            fold_statistics = {}

            print('processing fold #', i)
            # Prepare the validation data: data from partition # k
            pre_test = Pre_data[i * num_val_samples: (i + 1) * num_val_samples]
            mid_test = Mid_data[i * num_val_samples: (i + 1) * num_val_samples]
            suc_test = Suc_data[i * num_val_samples: (i + 1) * num_val_samples]
            c1_test = C1_data[i * num_val_samples: (i + 1) * num_val_samples]
            c2_test = C2_data[i * num_val_samples: (i + 1) * num_val_samples]
            y_test = Y_data[i * num_val_samples: (i + 1) * num_val_samples]

            # Prepare the training data: data from all other partitions
            pre_train = np.concatenate([Pre_data[:i * num_val_samples],Pre_data[(i + 1) * num_val_samples:]], axis=0)
            mid_train = np.concatenate([Mid_data[:i * num_val_samples], Mid_data[(i + 1) * num_val_samples:]], axis=0)
            suc_train = np.concatenate([Suc_data[:i * num_val_samples], Suc_data[(i + 1) * num_val_samples:]], axis=0)
            c1_train = np.concatenate([C1_data[:i * num_val_samples], C1_data[(i + 1) * num_val_samples:]], axis=0)
            c2_train = np.concatenate([C2_data[:i * num_val_samples], C2_data[(i + 1) * num_val_samples:]], axis=0)
            y_train = np.concatenate([Y_data[:i * num_val_samples],Y_data[(i + 1) * num_val_samples:]], axis=0)

            model = self.define_model(64, 1, 5,'relu', 'sigmoid', 0.5)
            model.fit([pre_train, mid_train, suc_train, c1_train, c2_train], y_train, epochs=20, batch_size=512)
            y_pred, y_true = self.predict(model,[pre_test, mid_test, suc_test, c1_test, c2_test], y_test)

            # Write the metrics for this fold.
            for label in self.data_model.label:
                fold_statistics[label] = {}
                f1 = f1_score(y_true, y_pred, average='micro', labels=[label])
                precision = precision_score(y_true, y_pred, average='macro', labels=[label])
                recall = recall_score(y_true, y_pred, average='micro', labels=[label])
                fold_statistics[label]['precision'] = precision
                fold_statistics[label]['recall'] = recall
                fold_statistics[label]['f1'] = f1

            # add averages
            fold_statistics['system'] = {}
            f1 = f1_score(y_true, y_pred, average='micro')
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='micro')
            fold_statistics['system']['precision'] = precision
            fold_statistics['system']['recall'] = recall
            fold_statistics['system']['f1'] = f1

            table_data = [[label,
                           format(fold_statistics[label]['precision'], ".3f"),
                           format(fold_statistics[label]['recall'], ".3f"),
                           format(fold_statistics[label]['f1'], ".3f")]
                          for label in self.data_model.label + ['system']]

            print(tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1'],
                                  tablefmt='orgtbl'))

            evaluation_statistics[i+1] = fold_statistics

        statistics_all_folds = {}

        for label in self.data_model.label + ['system']:
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
                      for label in self.data_model.label + ['system']]

        logging.info("\n"+tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
                       tablefmt='orgtbl'))
