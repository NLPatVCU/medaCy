from medacy.tools import DataFile, Annotations
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import os, logging, tempfile

class Simple_NN:

    def __init__(self, x_train = None, x_val = None, y_train = None, y_val = None, x_test = None, y_test = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test




