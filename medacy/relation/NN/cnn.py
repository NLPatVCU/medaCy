#Author : Samantha Mahendran for RelaCy

from .embedding import Embeddings
from medacy.relation.models import Model
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras import layers

import numpy as np

class CNN:
    def __init__(self):
        self.data_model = Model()

    def build_Model(self, output_classes, hidden_units = 32, filter_conv = 3, filter_maxPool = 5, hidden_activation = 'relu',
                    output_activation = 'softmax', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy']):
        model = Sequential()
        model.add(layers.Embedding(self.data_model.common_words, self.data_model.embedding_dim,
                                   input_length=self.data_model.maxlen))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation= hidden_activation))
        model.add(layers.MaxPooling1D(filter_maxPool))
        model.add(layers.Conv1D(hidden_units, filter_conv, activation= hidden_activation))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(output_classes, activation=output_activation))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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

    def evaluate_Model(self, model, x_test, y_test ):

        pred = model.predict(x_test)
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print ("Accuracy :", test_acc)
        print ("Loss : ", test_loss)
        print (classification_report(y_true, y_pred))
        matrix = confusion_matrix(y_true, y_pred)
        print (matrix)