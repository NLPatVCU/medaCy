#Author : Samantha Mahendran for RelaCy

from medacy.relation.models import Model
import os
import numpy as np

class Embeddings:

    def __init__(self, path, model, embedding_dim = 200):
        self.data_model = model
        self.path = path
        self.embedding_dim = embedding_dim
        self.build_embedding_layer()

    def read_embeddings_from_file(self):
        """
        Function to read external embedding files to build an index mapping words (as strings)
        to their vector representation (as number vectors).
        :return dictionary: word vectors
        """
        print("Reading external embedding file ......")
        if not os.path.isfile(self.path):
            raise FileNotFoundError("Not a valid file path")

        embeddings_index = {}
        with open(self.path) as f:
            next(f)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

        return embeddings_index

    def build_embedding_layer(self):

        embeddings_index = self.read_embeddings_from_file()
        self.embedding_matrix = np.zeros((self.data_model.common_words, self.embedding_dim))

        print("Building embedding matrix ....")
        for word, i in self.data_model.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < self.data_model.common_words:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    self.embedding_matrix[i] = embedding_vector

        print("Finished building embedding matrix ....")
        return self




