import logging, os
from tabulate import tabulate
from statistics import mean
from sklearn_crfsuite import metrics
from ..learn.stratified_k_fold import SequenceStratifiedKFold

from .base.base_model import BaseModel
from ..tools import DataLoader
from ..tools import model_to_ann

class CRFModel(BaseModel):


    def __init__(self, medacy_pipeline):
        super().__init__('CRF', medacy_pipeline)


    def fit(self, training_data_loader):
        """
            Runs training data through our pipeline and fits it using the CRF algorithm
            :param training_data_loader: Instance of DataLoader containing training files
            :return model: Trained model
        """

        assert isinstance(training_data_loader, DataLoader), "Must pass in an instance of DataLoader containing your training files"

        medacy_pipeline = self.pipeline
        nlp = medacy_pipeline.spacy_pipeline
        feature_extractor = medacy_pipeline.get_feature_extractor()

        # These arrays will store the sequences of features and sequences of corresponding labels
        X_data = []
        Y_data = []

        # Look into parallelizing as discussed in spacy.
        for data_file in training_data_loader.get_files():
            logging.info("Processing file: %s", data_file.file_name)

            with open(data_file.raw_path, 'r') as raw_text:
                doc = nlp.make_doc(raw_text.read())

            # Link ann_path to doc
            doc.set_extension('gold_annotation_file', default=data_file.ann_path, force=True)

            # Link metamapped file to doc for use in MetamapComponent if exists
            if training_data_loader.is_metamapped():
                doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

            # run 'er through
            doc = medacy_pipeline(doc)

            # The document has now been run through the pipeline. All annotations are overlayed - pull features.
            features, labels = feature_extractor(doc)

            X_data += features
            Y_data += labels

        logging.info("Feature Extraction Completed")
        # TODO Have the model defined in the Model class, not the pipeline. Have a pipeline store a list of models it supports.
        learner_name, learner = medacy_pipeline.get_learner()
        logging.info("Training: %s", learner_name)

        learner.fit(X_data, Y_data)
        logging.info("Successfully Trained: %s", learner_name)

        self.model = learner
        return self.model


    def predict(self, new_data_loader):
        """
        Predicts on the new data using the trained model, if model is not yet trained will return none.
        Outputs predictions to a /predictions directory where new example data is located.
        :param new_data_loader: DataLoader instance containing examples to predict on
        :return:
        """

        assert isinstance(new_data_loader, DataLoader), "Must pass in an instance of DataLoader containing your examples to be used for prediction"

        if self.model is None:
            return None
        else:
            model = self.model

        medacy_pipeline = self.pipeline

        # create directory to write predictions to
        prediction_directory = new_data_loader.data_directory + "/predictions/"
        if os.path.isdir(prediction_directory):
            logging.warning("Overwritting existing predictions")
        else:
            os.makedirs(prediction_directory)

        for data_file in new_data_loader.get_files():
            logging.info("Predicting file: %s", data_file.file_name)

            with open(data_file.raw_path, 'r') as raw_text:
                doc = medacy_pipeline.spacy_pipeline.make_doc(raw_text.read())

            if data_file.metamapped_path is not None:
                doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

            # run through the pipeline
            doc = medacy_pipeline(doc)

            ann_file_contents = model_to_ann(model, medacy_pipeline, doc)
            with open(prediction_directory+data_file.file_name+".ann", "a+") as f:
                f.write(ann_file_contents)

    # TODO untested after transfer from experimental codebase, should work though.
    def cross_validate(self, training_data_loader, num_folds=10):
        """
        Performs k-fold stratified cross-validation using our model and pipeline.
        :param training_data_loader:
        :param num_folds:
        :return: Prints out performance metrics
        """

        assert isinstance(training_data_loader, DataLoader), "Must pass in an instance of DataLoader containing your training files"
        assert self.model is not None, "Model is not yet trained, cannot cross validate."
        assert num_folds > 0, "Number of folds for cross validation must be greater than 0"

        medacy_pipeline = self.pipeline
        nlp = medacy_pipeline.spacy_pipeline
        feature_extractor = medacy_pipeline.get_feature_extractor()

        # These arrays will store the sequences of features and sequences of corresponding labels
        X_data = []
        Y_data = []

        # Look into parallelizing as discussed in spacy.
        # Run data through our pipeline and extract features
        for data_file in training_data_loader.get_files():
            logging.info("Processing file: %s", data_file.file_name)

            with open(data_file.raw_path, 'r') as raw_text:
                doc = nlp.make_doc(raw_text.read())

            # Link ann_path to doc
            doc.set_extension('gold_annotation_file', default=data_file.ann_path, force=True)

            # Link metamapped file to doc for use in MetamapComponent if exists
            if training_data_loader.is_metamapped():
                doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

            # run 'er through
            doc = medacy_pipeline(doc)

            # The document has now been run through the pipeline. All annotations are overlayed - pull features.
            features, labels = feature_extractor(doc)

            X_data += features
            Y_data += labels

        cv = SequenceStratifiedKFold(folds=num_folds)

        named_entities = medacy_pipeline.entities

        evaluation_statistics = {}
        fold = 1
        for train_indices, test_indices in cv(X_data, Y_data):
            # TODO Have the model defined in the Model class, not the pipeline. Have a pipeline store a list of models it supports.
            fold_statistics = {}
            learner_name, learner = medacy_pipeline.get_learner()

            X_train = [X_data[index] for index in train_indices]
            y_train = [Y_data[index] for index in train_indices]

            X_test = [X_data[index] for index in test_indices]
            y_test = [Y_data[index] for index in test_indices]

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