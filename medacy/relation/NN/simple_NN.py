#Author : Samantha Mahendran for RelaCy

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from keras import models
from keras import layers
import numpy as np

import os, logging, tempfile

class Simple_NN:

    def __init__(self, model):
        self.data_model = model
        # self.embed = Embeddings()

    def build_Model(self, output_classes, train_data, hidden_units = 64, hidden_activation = 'relu',
                    output_activation = 'softmax', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'] ):

        model = models.Sequential()
        model.add(layers.Dense(hidden_units, activation= hidden_activation, input_shape=(train_data.shape[1],)))
        # model.add(layers.Dense(hidden_units, activation= hidden_activation))
        model.add(layers.Dense(output_classes, activation= output_activation))

        model.compile(optimizer= optimizer,loss= loss, metrics= metrics)

        return model

    def build_Embedding_Model(self, output_classes, hidden_units = 64, hidden_activation = 'relu',
                    output_activation = 'softmax', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'] ):

        model = models.Sequential()
        model.add(layers.Embedding(self.data_model.common_words, self.data_model.embedding_dim, input_length=self.data_model.maxlen))
        model.add(layers.Flatten())
        model.add(layers.Dense(hidden_units, activation= hidden_activation))
        model.add(layers.Dense(hidden_units, activation=hidden_activation))
        model.add(layers.Dense(output_classes, activation= output_activation))

        model.compile(optimizer= optimizer,loss= loss, metrics= metrics)

        return model

    # def build_external_Embedding_Model(self, output_classes, hidden_units = 64, hidden_activation = 'relu',
    #                 output_activation = 'softmax', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'] ):
    #
    #     model = models.Sequential()
    #     model.add(layers.Embedding(self.data_model.common_words, self.data_model.embedding_dim, input_length=self.data_model.maxlen))
    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(hidden_units, activation= hidden_activation))
    #     model.add(layers.Dense(output_classes, activation= output_activation))
    #
    #     model.layers[0].set_weights([self.embed.embedding_matrix])
    #     model.layers[0].trainable = False
    #
    #     model.compile(optimizer= optimizer,loss= loss, metrics= metrics)
    #
    #     return model

    def fit_Model(self, model, x_train, y_train, no_epochs = 20, batch_Size = 512, validation = None):

        history = model.fit(x_train, y_train, epochs= no_epochs, batch_size= batch_Size, validation_data=validation)
        # history_dict = history.history
        # print (history_dict.keys())
        loss = history.history['loss']
        acc = history.history['acc']
        if validation is not None:
            val_loss = history.history['val_loss']
            val_acc = history.history['val_acc']
            max_epoch = val_acc.index(max(val_acc))+1
            # max_epoch = val_loss.index(min(val_loss)) + 1
            # print('Optimum epochs : ',max_epoch)
            self.data_model.plot_graphs(loss, val_loss, 'Epochs','Loss','Training loss', 'Validation loss','Training and validation loss' )
            self.data_model.plot_graphs(acc, val_acc, 'Epochs', 'Acc', 'Training acc', 'Validation acc','Training and validation acc')
            return model, loss, val_loss, acc, val_acc, max_epoch

        return model, loss, acc

    def evaluate_Model(self, model, x_test, y_test ):
        pred = model.predict(x_test)
        y_true = self.data_model.encoder.classes_[np.argmax(pred, axis=1)]
        y_pred = self.data_model.encoder.classes_[np.argmax(y_test, axis=1)]
        print(y_pred)
        # y_pred = np.argmax(pred, axis=1)
        # y_true = np.argmax(y_test, axis=1)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print ("Accuracy :", test_acc)
        print ("Loss : ", test_loss)
        print("Missing labels", set(y_true) - set(y_pred))
        print (classification_report(y_true, y_pred))
        print(f1_score(y_true, y_pred, average='micro'))
        # print (classification_report(y_true, y_pred, labels=self.data_model.label))
        matrix = confusion_matrix(y_true, y_pred)
        print (matrix)


