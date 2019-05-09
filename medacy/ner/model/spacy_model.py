from os import listdir, makedirs
from os.path import isfile, join, isdir
import random
from datetime import datetime
import logging
import joblib
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from medacy.tools import Annotations, DataFile
from medacy.data import Dataset

class SpacyModel:
    model = None

    def __init__(self, model=None):
        self.model = model

    def fit(self, dataset, spacy_model_name, new_model_name=None, iterations=30):
        """ Train a spaCy model using a medaCy dataset. Can be new or continued training.

        :param dataset: medaCy dataset object. Contain .ann and .txt files
        :param spacy_model_name: String of the spaCy model to use
        :param new_model_name: What the new model will be saved as. Defaults to a timestamp
        :param iterations: Training iterations to do
        :return model: A trained spaCy instance
        """
        data_files = dataset.get_data_files()
        labels = dataset.get_labels()
        train_data = dataset.get_training_data()

        print('New labels:')
        print(labels)

        if new_model_name is None:
            timestamp = datetime.now().timestamp()
            new_model_name = datetime.utcfromtimestamp(timestamp).strftime('%m-%d-%Y-%H%M%S')

        # Set up the pipeline and entity recognizer, and train the new entity.
        random.seed(0)
        if spacy_model_name is not None:
            nlp = spacy.load(spacy_model_name)  # load existing spaCy model
            print("Loaded model '%s'" % spacy_model_name)
        else:
            nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")
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

        for label in labels:
            ner.add_label(label)

        if spacy_model_name is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()
        move_names = list(ner.move_names)
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):  # only train NER
            sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            for itn in range(iterations):
                random.shuffle(train_data)
                batches = minibatch(train_data, size=sizes)
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                print("Losses", losses)

        nlp.meta["name"] = new_model_name  # rename model
        self.model = nlp

        return nlp

    def predict(self, dataset, prediction_directory=None):
        """
        Generates predictions over a string or a dataset utilizing the pipeline equipped to the
        instance.

        :param documents: a string or Dataset to predict
        :param prediction_directory: the directory to write predictions if doing bulk prediction
                                     (default: */prediction* sub-directory of Dataset)
        :return:
        """
        if not isinstance(dataset, (Dataset, str)):
            raise TypeError("Must pass in an instance of Dataset")
        if self.model is None:
            raise ValueError("must fit or load a pickled model before predicting")

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

                prediction_filename = join(prediction_directory, data_file.file_name + ".txt")
                logging.debug("Writing to: %s", prediction_filename)

                with open(prediction_filename, 'w') as prediction_file:
                    print(predictions, file=prediction_file)

        if isinstance(dataset, str):
            doc = nlp(dataset)
            entities = []
            for ent in doc.ents:
                entities.append((ent.label_, ent.text))
            return entities

    def load(self, path):
        """
        Loads a pickled model.

        :param path: File path to directory where fitted model should be dumped
        :return:
        """
        self.model = joblib.load(path)

    def dump(self, path):
        """
        Dumps a model into a pickle file

        :param path: Directory path to dump the model
        :return:
        """
        assert self.model is not None, "Must fit model before dumping."
        joblib.dump(self.model, path)