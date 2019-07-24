#Author : Samantha Mahendran for RelaCy

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os, logging, tempfile

class Evaluation:

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