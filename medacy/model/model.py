import importlib
import logging
import os
from itertools import cycle
from pathlib import Path
from shutil import copyfile
from statistics import mean
from typing import List, Tuple, Dict, Iterable

import joblib
import numpy as np
from sklearn_crfsuite import metrics
from tabulate import tabulate

from medacy.data.annotations import Annotations, EntTuple
from medacy.data.dataset import Dataset
from medacy.pipeline_components.feature_extractors import FeatureTuple
from medacy.pipelines.base.base_pipeline import BasePipeline

DEFAULT_NUM_FOLDS = 10


def create_folds(y, num_folds=DEFAULT_NUM_FOLDS) -> List[Tuple[FeatureTuple, List]]:
    """
    Partitions a data set of sequence labels and classifications into a number of stratified folds. Each partition
    should have an evenly distributed representation of sequence labels. Without stratification, under-representated
    labels may not appear in some folds. Returns an iterable [(X*,y*), ...] where each element contains the indices
    of the train and test set for the particular testing fold.

    See Dietterich, 1997 "Approximate Statistical Tests for Comparing Supervised Classification
    Algorithms" for in-depth analysis.

    :param y: a collection of sequence labels
    :param num_folds: the number of folds (defaults to five, but must be >= 2
    :return: an iterable
    """
    if not isinstance(num_folds, int) or num_folds < 2:
        raise ValueError(f"'num_folds' must be an int >= 2, but is {repr(num_folds)}")

    # labels are ordered by most examples in data
    labels = np.unique([label for sequence in y for label in sequence])
    np.flip(labels)

    added = np.ones(len(y), dtype=bool)
    partitions = [[] for _ in range(num_folds)]
    partition_cycler = cycle(partitions)

    for label in labels:
        possible_sequences = [index for index, sequence in enumerate(y) if label in sequence]
        for index in possible_sequences:
            if added[index]:
                partition = next(partition_cycler)
                partition.append(index)
                added[index] = 0

    train_test_array = []

    for i, y in enumerate(partitions):
        X = []
        for j, partition in enumerate(partitions):
            if i != j:
                X += partition

        train_test_array.append((X, y))

    return train_test_array


def sequence_to_ann(X: List[FeatureTuple], y: List[str], file_names: Iterable[str]) -> Dict[str, Annotations]:
    """
    Creates a dictionary of document-level Annotations objects for a given sequence
    :param X: A list of sentence level zipped (features, indices, document_name) tuples
    :param y: A  list of sentence-level lists of tags
    :param file_names: A list of file names that are used by these sequences
    :return: A dictionary mapping txt file names (the whole path) to their Annotations objects, where the
    Annotations are constructed from the X and y data given here.
    """
    # Flattening nested structures into 2d lists
    anns = {filename: Annotations([]) for filename in file_names}
    tuples_by_doc = {filename: [] for filename in file_names}
    document_indices = []
    span_indices = []

    for sequence in X:
        document_indices += [sequence.file_name] * len(sequence.features)
        span_indices.extend(sequence.indices)

    groundtruth = [element for sentence in y for element in sentence]

    # Map the predicted sequences to their corresponding documents
    i = 0

    while i < len(groundtruth):
        if groundtruth[i] == 'O':
            i += 1
            continue

        entity = groundtruth[i]
        document = document_indices[i]
        first_start, first_end = span_indices[i]
        # Ensure that consecutive tokens with the same label are merged
        while i < len(groundtruth) - 1 and groundtruth[i + 1] == entity:  # If inside entity, keep incrementing
            i += 1

        last_start, last_end = span_indices[i]
        tuples_by_doc[document].append((entity, first_start, last_end))
        i += 1

    # Create the Annotations objects
    for file_name, tups in tuples_by_doc.items():
        ann_tups = []
        with open(file_name) as f:
            text = f.read()
        for tup in tups:
            entity, start, end = tup
            ent_text = text[start:end]
            new_tup = EntTuple(entity, start, end, ent_text)
            ann_tups.append(new_tup)
        anns[file_name].annotations = ann_tups

    return anns


def write_ann_dicts(output_dir: Path, dict_list: List[Dict[str, Annotations]]) -> Dict[str, Annotations]:
    """
    Merges a list of dicts of Annotations into one dict representing all the individual ann files and prints the
    ann data for both the individual Annotations and the combined one.
    :param output_dir: Path object of the output directory (a subdirectory is made for each fold)
    :param dict_list: a list of file_name: Annotations dictionaries
    :return: The merged Annotations dict, if wanted
    """
    file_names = set()
    for d in dict_list:
        file_names |= set(d.keys())

    all_annotations_dict = {filename: Annotations([]) for filename in file_names}
    for i, fold_dict in enumerate(dict_list, 1):
        fold_dir = output_dir / f"fold_{i}"
        os.mkdir(fold_dir)
        for file_name, ann in fold_dict.items():
            # Write the Annotations from the individual fold to file;
            # Note that in this is written to the fold_dir, which is a subfolder of the output_dir
            ann.to_ann(fold_dir / (os.path.basename(file_name).rstrip("txt") + "ann"))
            # Merge the Annotations from the fold into the inter-fold Annotations
            all_annotations_dict[file_name] |= ann

    # Write the Annotations that are the combination of all folds to file
    for file_name, ann in all_annotations_dict.items():
        output_file_path = output_dir / (os.path.basename(file_name).rstrip("txt") + "ann")
        ann.to_ann(output_file_path)

    return all_annotations_dict


class Model:
    """
    A medaCy Model allows the fitting of a named entity recognition model to a given dataset according to the
    configuration of a given medaCy pipeline. Once fitted, Model instances can be used to predict over documents.
    Also included is a function for cross validating over a dataset for measuring the performance of a pipeline.

    :ivar pipeline: a medaCy pipeline, must be a subclass of BasePipeline (see medacy.pipelines.base.BasePipeline)
    :ivar model: weights, if the model has been fitted
    :ivar X_data: X_data from the pipeline; primarily for internal use
    :ivar y_data: y_data from the pipeline; primarily for internal use
    """

    def __init__(self, medacy_pipeline, model=None):

        if not isinstance(medacy_pipeline, BasePipeline):
            raise TypeError("Pipeline must be a medaCy pipeline that interfaces medacy.pipelines.base.BasePipeline")

        self.pipeline = medacy_pipeline
        self.model = model

        # These arrays will store the sequences of features and sequences of corresponding labels
        self.X_data = []
        self.y_data = []

        # Run an initializing document through the pipeline to register all token extensions.
        # This allows the gathering of pipeline information prior to fitting with live data.
        doc = self.pipeline(medacy_pipeline.spacy_pipeline.make_doc("Initialize"), predict=True)
        if doc is None:
            raise IOError("Model could not be initialized with the set pipeline.")

    def preprocess(self, dataset):
        """
        Preprocess dataset into a list of sequences and tags.
        :param dataset: Dataset object to preprocess.
        """
        self.X_data = []
        self.y_data = []
        # Run all Docs through the pipeline before extracting features, allowing for pipeline components
        # that require inter-dependent doc objects
        docs = [self._run_through_pipeline(data_file) for data_file in dataset if data_file.txt_path]
        for doc in docs:
            features, labels = self._extract_features(doc)
            self.X_data += features
            self.y_data += labels

    def fit(self, dataset: Dataset, groundtruth_directory: Path = None):
        """
        Runs dataset through the designated pipeline, extracts features, and fits a conditional random field.
        :param dataset: Instance of Dataset.
        :return model: a trained instance of a sklearn_crfsuite.CRF model.
        """

        groundtruth_directory = Path(groundtruth_directory) if groundtruth_directory else False

        report = self.pipeline.get_report()
        self.preprocess(dataset)

        if groundtruth_directory:
            logging.info(f"Writing dataset groundtruth to {groundtruth_directory}")
            for file_path, ann in sequence_to_ann(self.X_data, self.y_data, {x[2] for x in self.X_data}).items():
                ann.to_ann(groundtruth_directory / (os.path.basename(file_path).strip("txt") + "ann"))

        learner_name, learner = self.pipeline.get_learner()
        logging.info(f"Training: {learner_name}")

        train_data = [x[0] for x in self.X_data]
        learner.fit(train_data, self.y_data)
        logging.info(f"Successfully Trained: {learner_name}\n{report}")

        self.model = learner
        return self.model

    def _predict_document(self, doc):
        """
        Generates an dictionary of predictions of the given model over the corresponding document. The passed document
        is assumed to be annotated by the same pipeline utilized when training the model.
        :param doc: A spacy document
        :return: an Annotations object containing the model predictions
        """

        feature_extractor = self.pipeline.get_feature_extractor()

        features, indices = feature_extractor.get_features_with_span_indices(doc)
        predictions = self.model.predict(features)
        predictions = [element for sentence in predictions for element in sentence]  # flatten 2d list
        span_indices = [element for sentence in indices for element in sentence]  # parallel array containing indices
        annotations = []

        i = 0
        while i < len(predictions):
            if predictions[i] == 'O':
                i += 1
                continue

            entity = predictions[i]
            first_start, first_end = span_indices[i]

            # Ensure that consecutive tokens with the same label are merged
            while i < len(predictions) - 1 and predictions[i + 1] == entity:  # If inside entity, keep incrementing
                i += 1

            last_start, last_end = span_indices[i]
            labeled_text = doc.text[first_start:last_end]
            new_ent = EntTuple(entity, first_start, last_end, labeled_text)
            annotations.append(new_ent)

            logging.debug(f"{doc._.file_name}: Predicted {entity} at ({first_start}, {last_end}) {labeled_text}")

            i += 1

        return Annotations(annotations)

    def predict(self, input_data, prediction_directory=None):
        """
        Generates predictions over a string or a input_data utilizing the pipeline equipped to the instance.

        :param input_data: a string, Dataset, or directory path to predict over
        :param prediction_directory: The directory to write predictions if doing bulk prediction
            (default: */prediction* sub-directory of Dataset)
        :return: if input_data is a str, returns an Annotations of the predictions;
            if input_data is a Dataset or a valid directory path, returns a Dataset of the predictions.

        Note that if input_data is supposed to be a directory path but the directory is not found, it will be predicted
        over as a string. This can be prevented by validating inputs with os.path.isdir().
        """

        if self.model is None:
            raise RuntimeError("Must fit or load a pickled model before predicting")

        if isinstance(input_data, str) and not os.path.isdir(input_data):
            doc = self.pipeline.spacy_pipeline.make_doc(input_data)
            doc.set_extension('file_name', default=None, force=True)
            doc._.file_name = 'STRING_INPUT'
            doc = self.pipeline(doc, predict=True)
            annotations = self._predict_document(doc)
            return annotations

        if isinstance(input_data, Dataset):
            input_files = [d.txt_path for d in input_data]
            # Change input_data to point to the Dataset's directory path so that we can use it
            # to create the prediction directory
            input_data = input_data.data_directory
        elif os.path.isdir(input_data):
            input_files = [os.path.join(input_data, f) for f in os.listdir(input_data) if f.endswith('.txt')]
        else:
            raise ValueError(f"'input_data' must be a string (which can be a directory path) or a Dataset, but is {repr(input_data)}")

        if prediction_directory is None:
            prediction_directory = os.path.join(input_data, 'predictions')
            if os.path.isdir(prediction_directory):
                logging.warning("Overwriting existing predictions at %s", prediction_directory)
            else:
                os.mkdir(prediction_directory)

        for file_path in input_files:
            file_name = os.path.basename(file_path).strip('.txt')
            logging.info("Predicting file: %s", file_path)

            with open(file_path, 'r') as f:
                doc = self.pipeline.spacy_pipeline.make_doc(f.read())

            doc.set_extension('file_name', default=None, force=True)
            doc._.file_name = file_name

            # run through the pipeline
            doc = self.pipeline(doc, predict=True)

            # Predict, creating a new Annotations object
            annotations = self._predict_document(doc)
            logging.debug("Writing to: %s", os.path.join(prediction_directory, file_name + ".ann"))
            annotations.to_ann(write_location=os.path.join(prediction_directory, file_name + ".ann"))

            # Copy the txt file so that the output will also be a Dataset
            copyfile(file_path, os.path.join(prediction_directory, file_name + ".txt"))

        return Dataset(prediction_directory)

    def cross_validate(self, training_dataset, num_folds=DEFAULT_NUM_FOLDS, prediction_directory=None, groundtruth_directory=None):
        """
        Performs k-fold stratified cross-validation using our model and pipeline.

        If the training dataset, groundtruth_directory and prediction_directory are passed, intermediate predictions during cross validation
        are written to the directory `write_predictions`. This allows one to construct a confusion matrix or to compute
        the prediction ambiguity with the methods present in the Dataset class to support pipeline development without
        a designated evaluation set.

        :param training_dataset: Dataset that is being cross validated
        :param num_folds: number of folds to split training data into for cross validation, defaults to 5
        :param prediction_directory: directory to write predictions of cross validation to
        :param groundtruth_directory: directory to write the ground truth MedaCy evaluates on
        :return: Prints out performance metrics, if prediction_directory
        """

        if num_folds <= 1:
            raise ValueError("Number of folds for cross validation must be greater than 1, but is %s" % repr(num_folds))

        groundtruth_directory = Path(groundtruth_directory) if groundtruth_directory else False
        prediction_directory = Path(prediction_directory) if prediction_directory else False

        for d in [groundtruth_directory, prediction_directory]:
            if d and not d.exists():
                raise NotADirectoryError(f"Options groundtruth_directory and predictions_directory must be existing directories, but one is {d}")

        pipeline_report = self.pipeline.get_report()

        self.preprocess(training_dataset)

        if not (self.X_data and self.y_data):
            raise RuntimeError("Must have features and labels extracted for cross validation")

        tags = sorted(self.pipeline.entities)
        logging.info(f'Tagset: {tags}')

        eval_stats = {}

        # Dict for storing mapping of sequences to their corresponding file
        fold_groundtruth_dicts = []
        fold_prediction_dicts = []
        file_names = {x.file_name for x in self.X_data}

        folds = create_folds(self.y_data, num_folds)

        for fold_num, fold_data in enumerate(folds, 1):
            train_indices, test_indices = fold_data
            fold_statistics = {}
            learner_name, learner = self.pipeline.get_learner()

            X_train = [self.X_data[index] for index in train_indices]
            y_train = [self.y_data[index] for index in train_indices]

            X_test = [self.X_data[index] for index in test_indices]
            y_test = [self.y_data[index] for index in test_indices]

            logging.info("Training Fold %i", fold_num)
            train_data = [x[0] for x in X_train]
            test_data = [x[0] for x in X_test]
            learner.fit(train_data, y_train)
            y_pred = learner.predict(test_data)

            if groundtruth_directory is not None:
                ann_dict = sequence_to_ann(X_test, y_test, file_names)
                fold_groundtruth_dicts.append(ann_dict)

            if prediction_directory is not None:
                ann_dict = sequence_to_ann(X_test, y_pred, file_names)
                fold_prediction_dicts.append(ann_dict)

            # Write the metrics for this fold.
            for label in tags:
                fold_statistics[label] = {
                    "recall": metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=[label]),
                    "precision": metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=[label]),
                    "f1": metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=[label])
                }

            # add averages
            fold_statistics['system'] = {
                "recall": metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=tags),
                "precision": metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=tags),
                "f1": metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=tags)
            }

            table_data = [
                [label,
                 format(fold_statistics[label]['precision'], ".3f"),
                 format(fold_statistics[label]['recall'], ".3f"),
                 format(fold_statistics[label]['f1'], ".3f")
                 ] for label in tags + ['system']
            ]

            logging.info('\n' + tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1'], tablefmt='orgtbl'))

            eval_stats[fold_num] = fold_statistics

        statistics_all_folds = {}

        for label in tags + ['system']:
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

        entity_counts = training_dataset.compute_counts()
        entity_counts['system'] = sum(v for k, v in entity_counts.items() if k in self.pipeline.entities)

        table_data = [
            [f"{label} ({entity_counts[label]})",  # Entity (Count)
             format(statistics_all_folds[label]['precision_average'], ".3f"),
             format(statistics_all_folds[label]['recall_average'], ".3f"),
             format(statistics_all_folds[label]['f1_average'], ".3f"),
             format(statistics_all_folds[label]['f1_min'], ".3f"),
             format(statistics_all_folds[label]['f1_max'], ".3f")
             ] for label in tags + ['system']
        ]

        # Combine the pipeline report and the resulting data, then log it or print it (whichever ensures that it prints)

        output_str = '\n' + pipeline_report + '\n\n' + tabulate(
            table_data,
            headers=['Entity (Count)', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
            tablefmt='orgtbl'
        )

        if logging.root.level > logging.INFO:
            print(output_str)
        else:
            logging.info(output_str)

        # Write groundtruth and predictions to file
        if groundtruth_directory:
            write_ann_dicts(groundtruth_directory, fold_groundtruth_dicts)
        if prediction_directory:
            write_ann_dicts(prediction_directory, fold_prediction_dicts)

        return statistics_all_folds

    def _run_through_pipeline(self, data_file):
        """
        Runs a DataFile through the pipeline, returning the resulting Doc object
        :param data_file: instance of DataFile
        :return: a Doc object
        """
        nlp = self.pipeline.spacy_pipeline
        logging.info("Processing file: %s", data_file.file_name)

        with open(data_file.txt_path, 'r', encoding='utf-8') as f:
            doc = nlp.make_doc(f.read())

        # Link ann_path to doc
        doc.set_extension('gold_annotation_file', default=None, force=True)
        doc.set_extension('file_name', default=None, force=True)

        doc._.gold_annotation_file = data_file.ann_path
        doc._.file_name = data_file.txt_path

        # run 'er through
        return self.pipeline(doc)

    def _extract_features(self, doc):
        """
        Extracts features from a Doc
        :param doc: an instance of Doc
        :return: a tuple of the feature dict and label list
        """

        feature_extractor = self.pipeline.get_feature_extractor()
        features, labels = feature_extractor(doc)

        logging.info(f"{doc._.file_name}: Feature Extraction Completed (num_sequences={len(labels)})")
        return features, labels

    def load(self, path):
        """
        Loads a pickled model.

        :param path: File path to directory where fitted model should be dumped
        :return:
        """
        model_name, model = self.pipeline.get_learner()

        if model_name == 'BiLSTM+CRF' or model_name == 'BERT':
            model.load(path)
            self.model = model
        else:
            self.model = joblib.load(path)

    def dump(self, path):
        """
        Dumps a model into a pickle file

        :param path: Directory path to dump the model
        :return:
        """
        if self.model is None:
            raise RuntimeError("Must fit model before dumping.")

        model_name, _ = self.pipeline.get_learner()

        if model_name == 'BiLSTM+CRF' or model_name == 'BERT':
            self.model.save(path)
        else:
            joblib.dump(self.model, path)

    @staticmethod
    def load_external(package_name):
        """
        Loads an external medaCy compatible Model. Require's the models package to be installed
        Alternatively, you can import the package directly and call it's .load() method.

        :param package_name: the package name of the model
        :return: an instance of Model that is configured and loaded - ready for prediction.
        """
        if importlib.util.find_spec(package_name) is None:
            raise ImportError("Package not installed: %s" % package_name)
        return importlib.import_module(package_name).load()
