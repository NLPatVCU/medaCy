import logging
import random
from os import makedirs
from os.path import join, isdir
from statistics import mean

import spacy
from sklearn_crfsuite import metrics
from spacy.gold import biluo_tags_from_offsets
from spacy.util import minibatch, compounding
from tabulate import tabulate

from medacy.data.dataset import Dataset
from medacy.data.annotations import Annotations
from medacy.model._model import construct_annotations_from_tuples
from medacy.model.stratified_k_fold import SequenceStratifiedKFold


class SpacyModel:
    """
    SpacyModel consists of convenience wrapper functions for pre-existing spaCy functionality.
    Attributes:
        model (spacy.Language): Trained model as spaCy language(). Usually appears as 'nlp' in
                                spaCy documentation.
    """
    model = None
    nlp = None
    new = True

    def __init__(self, spacy_model_name=None, cuda=-1):
        if cuda >= 0:
            spacy.prefer_gpu()

        if spacy_model_name is None:
            self.nlp = spacy.blank("en")  # create blank Language class
            logging.info("Created blank 'en' model")
        else:
            self.nlp = spacy.load(spacy_model_name)  # load existing spaCy model
            self.new = False
            logging.info("\nLoaded model '%s'", spacy_model_name)

    def fit(self, dataset, iterations=20, asynchronous=False, labels=None):
        """ Train a spaCy model using a medaCy dataset. Can be new or continued training.

        :param dataset: medaCy dataset object
        :param spacy_model_name: String of the spaCy model name to use
        :param iterations: Training iterations to do
        :return nlp: A trained spaCy model (language)
        """
        train_data = dataset.get_training_data()
        labels = sorted(list(dataset.get_labels()))
        logging.info('Labels: %s', labels)

        logging.info('Fitting new model...\n')

        # Set up the pipeline and entity recognizer, and train the new entity.
        random.seed(0)
        nlp = self.nlp

        # Add entity recognizer to model if it's not in the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner)

        # otherwise, get it, so we can add labels to it
        else:
            ner = nlp.get_pipe("ner")
            logging.info('Original labels:')
            logging.info(ner.labels)

        for label in labels:
            ner.add_label(label)

        if self.new:
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
                logging.info("Losses %s", str(losses))

        self.model = nlp

        return nlp

    def predict(self, dataset, prediction_directory=None, groundtruth_directory=None):
        """
        Generates predictions over a string or a medaCy dataset

        :param dataset: a string or medaCy Dataset to predict
        :param prediction_directory: the directory to write predictions if doing bulk prediction
                                     (default: */prediction* sub-directory of Dataset)
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

                with open(data_file.get_text_path(), 'r') as source_text_file:
                    text = source_text_file.read()

                doc = nlp(text)

                predictions = []

                for ent in doc.ents:
                    predictions.append((ent.label_, ent.start_char, ent.end_char, ent.text))

                annotations = construct_annotations_from_tuples(text, predictions)

                prediction_filename = join(prediction_directory, data_file.file_name + ".ann")
                logging.debug("Writing to: %s", prediction_filename)
                annotations.to_ann(write_location=prediction_filename)

        if isinstance(dataset, str):
            doc = nlp(dataset)

            entities = []

            for ent in doc.ents:
                entities.append((ent.start_char, ent.end_char, ent.label_))

            return entities

    def cross_validate(self, num_folds=5, training_dataset=None, epochs=20, prediction_directory=None, groundtruth_directory=None, asynchronous=None):
        """
        Runs a cross validation.

        :param folds: Number of fold to do for the cross validation.
        :param training_dataset: Path to the directory of BRAT files to use for the training data.
        :param spacy_model_name: Name of the spaCy model to start from.
        :param epochs: Number of epochs to us for every fold training.
        """
        if num_folds <= 1:
            raise ValueError("Number of folds for cross validation must be greater than 1")

        if training_dataset is None:
            raise ValueError("Need a dataset to evaluate")

        train_data = training_dataset.get_training_data()

        labels = set()
        for document in train_data:
            for entity in document[1]['entities']:
                tag = entity[2]
                labels.add(tag)
        labels = list(labels)
        labels.sort()
        logging.info('Labels: %s', labels)

        x_data, y_data = zip(*train_data)

        skipped_files = []
        eval_stats = {}

        folds = SequenceStratifiedKFold(folds=num_folds)
        fold = 1

        for train_indices, test_indices in folds(x_data, y_data):
            logging.info("\n----EVALUATING FOLD %d----", fold)
            self.model = None
            fold_statistics = {}

            x_subdataset = training_dataset.get_subdataset(train_indices)
            self.fit(x_subdataset, iterations=epochs, labels=labels)
            logging.info('Done training!\n')

            nlp = self.model

            y_subdataset = training_dataset.get_subdataset(test_indices)

            y_test = []
            y_pred = []

            for ann in y_subdataset.generate_annotations():

                with open(ann.source_text_path, 'r') as source_text_file:
                    text = source_text_file.read()

                doc = nlp(text)

                # test_entities = annotations.get_entities(format='spacy')[1]['entities']
                test_entities = ann.get_entity_annotations(format='spacy')[1]['entities']
                test_entities = self.entities_to_biluo(doc, test_entities)
                y_test.append(test_entities)

                pred_entities = self.predict(text)
                pred_entities = self.entities_to_biluo(doc, pred_entities)
                y_pred.append(pred_entities)

            logging.debug('\n------y_test------')
            logging.debug(y_test)
            logging.debug('\n------y_pred------')
            logging.debug(y_pred)

            # Write the metrics for this fold.
            for label in labels:
                fold_statistics[label] = {
                    'recall': metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=[label]),
                    'precision': metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=[label]),
                    'f1': metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=[label])
                }

            # add averages
            fold_statistics['system'] = {
                'recall': metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=labels),
                'precision': metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=labels),
                'f1': metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
            }

            table_data = [
                [label,
                format(fold_statistics[label]['precision'], ".3f"),
                format(fold_statistics[label]['recall'], ".3f"),
                format(fold_statistics[label]['f1'], ".3f")
                ] for label in labels + ['system']
            ]

            logging.info('\n' + tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1'], tablefmt='orgtbl'))

            eval_stats[fold] = fold_statistics
            fold += 1

        if skipped_files:
            logging.info('\nWARNING. SKIPPED THE FOLLOWING ANNOTATIONS:')
            logging.info(skipped_files)

        statistics_all_folds = {}

        for label in labels + ['system']:
            statistics_all_folds[label] = {
                'precision_average': mean(eval_stats[fold][label]['precision'] for fold in eval_stats),
                'precision_max': max(eval_stats[fold][label]['precision'] for fold in eval_stats),
                'precision_min': min(eval_stats[fold][label]['precision'] for fold in eval_stats),
                'recall_average': mean(eval_stats[fold][label]['recall'] for fold in eval_stats),
                'recall_max': max(eval_stats[fold][label]['recall'] for fold in eval_stats),
                'f1_average': mean(eval_stats[fold][label]['f1'] for fold in eval_stats),
                'f1_max': max(eval_stats[fold][label]['f1'] for fold in eval_stats),
                'f1_min': min(eval_stats[fold][label]['f1'] for fold in eval_stats),
            }

        table_data = [
            [label,
            format(statistics_all_folds[label]['precision_average'], ".3f"),
            format(statistics_all_folds[label]['recall_average'], ".3f"),
            format(statistics_all_folds[label]['f1_average'], ".3f"),
            format(statistics_all_folds[label]['f1_min'], ".3f"),
            format(statistics_all_folds[label]['f1_max'], ".3f")
            ] for label in labels + ['system']
        ]

        table_string = '\n' + tabulate(
            table_data,
            headers=['Entity', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
            tablefmt='orgtbl'
        )
        logging.info(table_string)

    def load(self, path):
        """
        Loads a spaCy model and sets it as new self.model.

        :param path: Path to directory of spaCy model.
        """
        nlp = spacy.load(path)
        self.model = nlp

    def dump(self, path):
        """
        Saves the spacy model using the to_disk() function.
        https://spacy.io/usage/saving-loading#models

        :param path: Directory path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")

        self.model.to_disk(path)

    def entities_to_biluo(self, doc, entities):
        """
        Converts entity span tuples into a suitable BILUO format for metrics.

        :param doc: spaCy doc of original text
        :param entities: Tuples to be converted

        :returns: List of new BILUO tags
        """
        spacy_biluo = biluo_tags_from_offsets(doc, entities)
        medacy_biluo = []
        for tag in spacy_biluo:
            if tag != 'O':
                tag = tag[2:]
            medacy_biluo.append(tag)
        return medacy_biluo
