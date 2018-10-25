"""
Builds a model with a given medacy pipeline and dataset
"""
import logging
from tabulate import tabulate
from statistics import mean
from sklearn_crfsuite import metrics

from ..pipelines.base import BasePipeline
from ..tools import DataLoader
from ..pipeline_components import MetaMap
from .stratified_k_fold import SequenceStratifiedKFold

class Learner:
    def __init__(self, medacy_pipeline, data_loader):
        """

        :param medacy_pipeline: A sub-class of BasePipeline such as ClinicalPipeline
        :param data_loader: An instance of DataLoader
        :param metamap: an instance of metamap, that if present will cause the data_loader to directory.
        """


        assert isinstance(medacy_pipeline, BasePipeline), "Pipeline must be a medaCy pipeline that interfaces medacy.pipelines.base.BasePipeline"
        assert isinstance(data_loader, DataLoader), "Must give an instance of DataLoader into the Learner"

        self.medacy_pipeline = medacy_pipeline
        self.data_loader = data_loader

        self.nlp = medacy_pipeline.spacy_pipeline
        self.feature_extractor = medacy_pipeline.get_feature_extractor()

        self.X_data = []
        self.y_data = []
        self.model = None





    def train(self, cross_validation=False):
        """
        Trains an initialized Learner
        :param cross_validation: Whether to perform stratified cross validation of model. The model will be saved first.
        :return:
        """
        data_loader = self.data_loader
        medacy_pipeline = self.medacy_pipeline
        nlp = medacy_pipeline.spacy_pipeline


        #These arrays will store the sequences of features and sequences of corresponding labels


        #Look into parallelizing as discussed in spacy.
        for data_file in data_loader.get_files():
            logging.info("Processing file: %s", data_file.file_name)

            with open(data_file.raw_path, 'r') as raw_text:
                doc = nlp.make_doc(raw_text.read())

            #Link ann_path to doc
            doc.set_extension('gold_annotation_file', default=data_file.ann_path, force=True)

            #Link metamapped file to doc for use in MetamapComponent if exists
            if data_loader.is_metamapped():
                doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

            #run 'er through
            doc = medacy_pipeline(doc)

            #The document has now be run through the pipeline. All annotations are overlayed - pull features.
            features, labels = self.feature_extractor(doc)

            self.X_data += features
            self.y_data += labels


        logging.info("Feature Extraction Completed")
        learner_name, learner = medacy_pipeline.get_learner()
        logging.info("Training: %s", learner_name)

        learner.fit(self.X_data, self.y_data)
        logging.info("Successfully Trained: %s", learner_name)

        self.model = learner
        return self.model

    def cross_validate(self):
        #TODO untested after transfer from experimental codebase, should work though.
        """
        Performs cross validation on trained mode.

        This should really go in another class.
        :return: prints out cross validation metrics.
        """

        assert self.model is not None, "Model is not yet trained, cannot cross validate."

        cv = SequenceStratifiedKFold(folds=10)
        X_data = self.X_data
        y_data = self.y_data

        named_entities = self.medacy_pipeline.entities

        evaluation_statistics = {}
        fold = 1
        for train_indices, test_indices in cv(X_data, y_data):

            fold_statistics = {}
            learner_name, learner = self.medacy_pipeline.get_learner()

            X_train = [X_data[index] for index in train_indices]
            y_train = [y_data[index] for index in train_indices]

            X_test = [X_data[index] for index in test_indices]
            y_test = [y_data[index] for index in test_indices]

            logging.info("Training Fold %i", fold)
            learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)

            for label in named_entities:
                fold_statistics[label] = {}
                recall = metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=[label])
                precision = metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=[label])
                f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=[label])
                fold_statistics[label]['precision'] = precision
                fold_statistics[label]['recall'] = recall
                fold_statistics[label]['f1'] = f1

            # add averages
            fold_statistics['system'] = {}
            recall = metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=named_entities)
            precision = metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=named_entities)
            f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=named_entities)
            fold_statistics['system']['precision'] = precision
            fold_statistics['system']['recall'] = recall
            fold_statistics['system']['f1'] = f1

            evaluation_statistics[fold] = fold_statistics
            fold += 1

        statistics_all_folds = {}

        for label in named_entities + ['system']:
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
                      for label in named_entities + ['system']]

        print(tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
                       tablefmt='orgtbl'))


















