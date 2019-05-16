from os import listdir, makedirs
from os.path import isfile, join, isdir
import random
import logging
import joblib
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from medacy.tools import Annotations, DataFile
from medacy.data import Dataset
from .stratified_k_fold import SequenceStratifiedKFold
from ._model import construct_annotations_from_tuples

class SpacyModel:
    model = None

    def fit(self, dataset, spacy_model_name, iterations=30, revision_texts=None, prefer_gpu=False):
        """ Train a spaCy model using a medaCy dataset. Can be new or continued training.

        :param dataset: medaCy dataset object. Contain .ann and .txt files
        :param spacy_model_name: String of the spaCy model to use
        :param iterations: Training iterations to do
        :return model: A trained spaCy instance
        """
        data_files = dataset.get_data_files()
        labels = dataset.get_labels()
        train_data = dataset.get_training_data()

        print('New labels:')
        print(labels)

        # Set up the pipeline and entity recognizer, and train the new entity.
        random.seed(0)

        if prefer_gpu:
            spacy.prefer_gpu()

        if spacy_model_name is None:
            nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")
        else:
            nlp = spacy.load(spacy_model_name)  # load existing spaCy model
            print("\nLoaded model '%s'" % spacy_model_name)

        # Add entity recognizer to model if it's not in the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner)

        # otherwise, get it, so we can add labels to it
        else:
            ner = nlp.get_pipe("ner")
            print('Original labels:')
            print(ner.labels)
            print()

        for label in labels:
            ner.add_label(label)

        if spacy_model_name is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

        with nlp.disable_pipes(*other_pipes):  # only train NER
            sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            for _ in range(iterations):
                random.shuffle(train_data)
                batches = minibatch(train_data, size=sizes)
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                print("Losses", losses)

        self.model = nlp

        return nlp

    def predict(self, dataset, prediction_directory=None):
        """
        Generates predictions over a string or a dataset utilizing the pipeline equipped to the
        instance.

        :param dataset: a string or Dataset to predict
        :param prediction_directory: the directory to write predictions if doing bulk prediction
                                     (default: */prediction* sub-directory of Dataset)
        :return:
        """
        if not isinstance(dataset, (Dataset, str)):
            raise TypeError("Must pass in an instance of Dataset")
        if self.model is None:
            raise ValueError("Must fit or load a pickled model before predicting")

        nlp = self.model

        if isinstance(dataset, Dataset):
            if prediction_directory is None:
                prediction_directory = str(dataset.data_directory) + "/predictions/"

            if isdir(prediction_directory):
                logging.warning("Overwriting existing predictions")
            else:
                makedirs(prediction_directory)

            for data_file in dataset.get_data_files():
                logging.info("Predicting file: %s", data_file.file_name)

                with open(data_file.raw_path, 'r') as raw_text:
                    doc = nlp(raw_text.read())

                with open(data_file.get_text_path(), 'r') as source_text_file:
                    text = source_text_file.read()

                predictions = self.predict(text)

                annotations = construct_annotations_from_tuples(text, predictions)

                prediction_filename = join(prediction_directory, data_file.file_name + ".ann")
                logging.debug("Writing to: %s", prediction_filename)
                annotations.to_ann(write_location=prediction_filename)

        if isinstance(dataset, str):
            doc = nlp(dataset)
            entities = []

            for ent in doc.ents:
                entities.append((ent.label_, ent.start_char, ent.end_char, ent.text))

            return entities

    def cross_validate(self, num_folds=10, training_dataset=None, spacy_model_name=None):
        if num_folds <= 1:
            raise ValueError("Number of folds for cross validation must be greater than 1")

        if training_dataset is None:
            raise ValueError("Need a dataset to evaluate")

        train_data = training_dataset.get_training_data()

        X_data, Y_data = zip(*train_data)

        evaluation_statistics = {}

        cv = SequenceStratifiedKFold(folds=num_folds)
        fold = 1

        for train_indices, test_indices in cv(X_data, Y_data):
            print("\n----EVALUATING FOLD %d----" % fold)

            subdataset = training_dataset.get_subdataset(train_indices)
            self.fit(subdataset, spacy_model_name, 1)

            fold += 1


    def load(self, path, prefer_gpu=False):
        """
        Loads a spaCy model and sets it as new self.model.

        :param path: Path to directory of spaCy model.
        """
        if prefer_gpu:
            spacy.prefer_gpu()
        nlp = spacy.load(path)
        self.model = nlp

    def save(self, path):
        """
        Saves the spacy model using the to_disk() function.
        https://spacy.io/usage/saving-loading#models

        :param path: Directory path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")

        self.model.to_disk(path)
