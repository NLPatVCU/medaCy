from tabulate import tabulate
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import *
from keras.layers import *
from keras import layers
from statistics import mean
import numpy as np
import logging

class Segment_CNN:

    def __init__(self, model, embedding = False):
        self.data_model = model
        self.embedding = embedding

    # define the model
    def define_model(self):

        # channel 1
        inputs1 = Input(shape=(self.data_model.maxlen,))
        embedding1 = Embedding(self.data_model.common_words, self.embedding.embedding_dim )(inputs1)
        # embedding1 = Embedding(self.data_model.common_words, self.embedding.embedding_dim,self.embedding.embedding_matrix, False )(inputs1)
        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)

        # channel 2
        inputs2 = Input(shape=(self.data_model.maxlen,))
        embedding2 = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(inputs2)
        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)

        # channel 3
        inputs3 = Input(shape=(self.data_model.maxlen,))
        embedding3 = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(inputs3)
        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)

        # channel 4
        inputs4 = Input(shape=(self.data_model.maxlen,))
        embedding4 = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(inputs4)
        conv4 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling1D(pool_size=2)(drop4)
        flat4 = Flatten()(pool4)

        # channel 5
        inputs5 = Input(shape=(self.data_model.maxlen,))
        embedding5 = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(inputs5)
        conv5 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding5)
        drop5 = Dropout(0.5)(conv5)
        pool5 = MaxPooling1D(pool_size=2)(drop5)
        flat5 = Flatten()(pool5)

        # merge
        merged = concatenate([flat1, flat2, flat3, flat4, flat5])

        # interpretation
        dense1 = Dense(18, activation='relu')(merged)
        outputs = Dense(11, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5], outputs=outputs)

        # compile
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # summarize
        print(model.summary())
        return model


    def define_Embedding_Model(self, hidden_units = 64, filter_conv = 1, filter_maxPool = 5, hidden_activation = 'relu'):
        model = Sequential()
        model.add(layers.Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                   input_length=self.data_model.maxlen))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation=hidden_activation))
        model.add(layers.Dropout(0.5))
        model.add(layers.MaxPooling1D(filter_maxPool))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation=hidden_activation))
        model.add(layers.GlobalMaxPooling1D())

        print(model.summary())
        if self.embedding:
            model.layers[0].set_weights([self.embedding.embedding_matrix])
            model.layers[0].trainable = False

        return model

    def fit_Model(self, model, x_train, y_train, no_epochs = 20, batch_Size = 512, validation = None,
                  output_activation = 'sigmoid', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy']):
        model.add(layers.Dense(len(self.data_model.label), activation=output_activation))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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
            pre_train = np.concatenate([Pre_data[:i * num_val_samples],Pre_data[(i + 1) * num_val_samples:]],axis=0)
            mid_train = np.concatenate([Mid_data[:i * num_val_samples], Mid_data[(i + 1) * num_val_samples:]], axis=0)
            suc_train = np.concatenate([Suc_data[:i * num_val_samples], Suc_data[(i + 1) * num_val_samples:]], axis=0)
            c1_train = np.concatenate([C1_data[:i * num_val_samples], C1_data[(i + 1) * num_val_samples:]], axis=0)
            c2_train = np.concatenate([C2_data[:i * num_val_samples], C2_data[(i + 1) * num_val_samples:]], axis=0)
            y_train = np.concatenate([Y_data[:i * num_val_samples],Y_data[(i + 1) * num_val_samples:]],axis=0)

            # model = self.build_external_Embedding_Model()
            model = self.define_model()
            model.fit([pre_train, mid_train, suc_train, c1_train, c2_train], y_train, epochs=10, batch_size=16)
            # model, loss, acc = self.fit_Model (model, x_train, y_train)
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

        logging.info("\n"+tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
                       tablefmt='orgtbl'))
