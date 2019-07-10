"""
A medaCy named entity recognition model wraps together three functionalities
"""

import logging, os, joblib, time, importlib
from medacy.data import Dataset
from .stratified_k_fold import SequenceStratifiedKFold
from medacy.ner.pipelines import BasePipeline
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count
from ._model import predict_document, construct_annotations_from_tuples
from sklearn_crfsuite import metrics
from tabulate import tabulate
from statistics import mean


class Model:

    def __init__(self, medacy_pipeline=None, model=None, n_jobs=cpu_count()):

        if not isinstance(medacy_pipeline, BasePipeline):
            raise TypeError("Pipeline must be a medaCy pipeline that interfaces medacy.pipelines.base.BasePipeline")

        self.pipeline = medacy_pipeline
        self.model = model

        # These arrays will store the sequences of features and sequences of corresponding labels
        self.X_data = []
        self.y_data = []
        self.n_jobs = n_jobs

        # Run an initializing document through the pipeline to register all token extensions.
        # This allows the gathering of pipeline information prior to fitting with live data.
        doc = self.pipeline(medacy_pipeline.spacy_pipeline.make_doc("Initialize"), predict=True)
        if doc is None:
            raise IOError("Model could not be initialized with the set pipeline.")

    def preprocess(self, dataset):
        try:
            pool = Pool(nodes=self.n_jobs)

            results = [pool.apipe(self._extract_features, data_file, self.pipeline, dataset.is_metamapped())
                    for data_file in dataset.get_data_files()]

            while any([i.ready() is False for i in results]):
                time.sleep(1)

            for idx, i in enumerate(results):
                X, y = i.get()
                self.X_data += X
                self.y_data += y

        except TypeError as error:
            if str(error) == "can not serialize 'cupy.core.core.ndarray' object":
                logging.info('Ran into GPU error. Switching to synchronous preprocessing...')
                self.X_data == []
                self.y_data == []
                for data_file in dataset.get_data_files():
                    features, labels = self._extract_features(data_file, self.pipeline, dataset.is_metamapped())
                    self.X_data += features
                    self.y_data += labels


    def fit(self, dataset):
        """
        Runs dataset through the designated pipeline, extracts features, and fits a conditional random field.

        :param training_data_loader: Instance of Dataset.
        :return model: a trained instance of a sklearn_crfsuite.CRF model.
        """

        if not isinstance(dataset, Dataset):
            raise TypeError("Must pass in an instance of Dataset containing your training files")
        if not isinstance(self.pipeline, BasePipeline):
            raise TypeError("Model object must contain a medacy pipeline to pre-process data")

        self.preprocess(dataset)

        logging.info("Currently Waiting")

        learner_name, learner = self.pipeline.get_learner()
        logging.info("Training: %s", learner_name)

        assert self.X_data, "Training data is empty."

        train_data = [x[0] for x in self.X_data]
        learner.fit(train_data, self.y_data)
        logging.info("Successfully Trained: %s", learner_name)

        self.model = learner
        return self.model

    def predict(self, dataset, prediction_directory=None, groundtruth_directory=None):
        """
        Generates predictions over a string or a dataset utilizing the pipeline equipped to the instance.

        :param documents: a string or Dataset to predict
        :param prediction_directory: the directory to write predictions if doing bulk prediction (default: */prediction* sub-directory of Dataset)
        :return:
        """

        if not isinstance(dataset, (Dataset, str)):
            raise TypeError("Must pass in an instance of Dataset containing your examples to be used for prediction")
        if self.model is None:
            raise ValueError("Must fit or load a pickled model before predicting")

        model = self.model
        medacy_pipeline = self.pipeline

        if isinstance(dataset, Dataset):
            # create directory to write predictions to
            prediction_directory = self.create_annotation_directory(directory=prediction_directory, training_dataset=dataset, option="predictions")

            # create directory to write groundtruth to
            groundtruth_directory = self.create_annotation_directory(directory=groundtruth_directory, training_dataset=dataset, option="groundtruth")

            for data_file in dataset.get_data_files():
                logging.info("Predicting file: %s", data_file.file_name)
                with open(data_file.raw_path, 'r') as raw_text:
                    doc = medacy_pipeline.spacy_pipeline.make_doc(raw_text.read())
                    doc.set_extension('file_name', default=data_file.file_name, force=True)
                    if data_file.metamapped_path is not None:
                        doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

                # run through the pipeline
                doc = medacy_pipeline(doc, predict=True)

                annotations = predict_document(model, doc, medacy_pipeline)
                logging.debug("Writing to: %s", os.path.join(prediction_directory, data_file.file_name + ".ann"))
                annotations.to_ann(write_location=os.path.join(prediction_directory, data_file.file_name + ".ann"))

        if isinstance(dataset, str):
            assert 'metamap_annotator' not in self.pipeline.get_components(), \
                "Cannot currently predict on the fly when metamap_component is in pipeline."

            doc = medacy_pipeline.spacy_pipeline.make_doc(dataset)
            doc.set_extension('file_name', default="STRING_INPUT", force=True)
            doc = medacy_pipeline(doc, predict=True)
            annotations = predict_document(model, doc, medacy_pipeline)
            return annotations

    def cross_validate(self, num_folds=5, training_dataset=None, prediction_directory=None, groundtruth_directory=None):
        """
        Performs k-fold stratified cross-validation using our model and pipeline.

        If the training dataset, groundtruth_directory and prediction_directory are passed, intermediate predictions during cross validation
        are written to the directory `write_predictions`. This allows one to construct a confusion matrix or to compute
        the prediction ambiguity with the methods present in the Dataset class to support pipeline development without
        a designated evaluation set.

        :param num_folds: number of folds to split training data into for cross validation
        :param training_dataset: Dataset that is being cross validated (optional)
        :param prediction_directory: directory to write predictions of cross validation to or `True` for default predictions sub-directory.
        :param groundtruth_directory: directory to write the ground truth MedaCy evaluates on
        :return: Prints out performance metrics, if prediction_directory
        """

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        if prediction_directory is not None and training_dataset is None:
            raise ValueError("Cannot generate predictions during cross validation if training dataset is not given."
                             " Please pass the training dataset in the 'training_dataset' parameter.")
        if groundtruth_directory is not None and training_dataset is None:
            raise ValueError("Cannot generate groundtruth during cross validation if training dataset is not given."
                             " Please pass the training dataset in the 'training_dataset' parameter.")

        # assert self.model is not None, "Cannot cross validate a un-fit model"
        self.preprocess(training_dataset)
        assert self.X_data is not None and self.y_data is not None, \
            "Must have features and labels extracted for cross validation"

        X_data = self.X_data
        Y_data = self.y_data

        medacy_pipeline = self.pipeline

        cv = SequenceStratifiedKFold(folds=num_folds)

        named_entities = medacy_pipeline.entities

        evaluation_statistics = {}
        fold = 1
        # Dict for storing mapping of sequences to their corresponding file
        groundtruth_by_document = {filename: [] for filename in list(set([x[2] for x in X_data]))}
        preds_by_document = {filename: [] for filename in list(set([x[2] for x in X_data]))}

        for train_indices, test_indices in cv(X_data, Y_data):
            fold_statistics = {}
            learner_name, learner = medacy_pipeline.get_learner()

            X_train = [X_data[index] for index in train_indices]
            y_train = [Y_data[index] for index in train_indices]

            X_test = [X_data[index] for index in test_indices]
            y_test = [Y_data[index] for index in test_indices]

            logging.info("Training Fold %i", fold)
            train_data = [x[0] for x in X_train]
            test_data = [x[0] for x in X_test]
            learner.fit(train_data, y_train)
            y_pred = learner.predict(test_data)



            if groundtruth_directory is not None:
                # Dict for storing mapping of sequences to their corresponding file

                # Flattening nested structures into 2d lists
                document_indices = []
                span_indices = []
                for sequence in X_test:
                    document_indices += [sequence[2] for x in range(len(sequence[0]))]
                    span_indices += [element for element in sequence[1]]
                groundtruth = [element for sentence in y_test for element in sentence]

                # Map the predicted sequences to their corresponding documents
                i=0
                while i < len(groundtruth):
                    if groundtruth[i] == 'O':
                        i+=1
                        continue
                    entity = groundtruth[i]
                    document = document_indices[i]
                    first_start, first_end = span_indices[i]
                    # Ensure that consecutive tokens with the same label are merged
                    while i < len(groundtruth) - 1 and groundtruth[i + 1] == entity:  # If inside entity, keep incrementing
                        i += 1
                    last_start, last_end = span_indices[i]
                    groundtruth_by_document[document].append((entity, first_start, last_end))
                    i+=1
            if prediction_directory is not None:
                # Dict for storing mapping of sequences to their corresponding file

                # Flattening nested structures into 2d lists
                document_indices = []
                span_indices = []
                for sequence in X_test:
                    document_indices += [sequence[2] for x in range(len(sequence[0]))]
                    span_indices += [element for element in sequence[1]]
                predictions = [element for sentence in y_pred for element in sentence]

                # Map the predicted sequences to their corresponding documents
                i=0
                while i < len(predictions):
                    if predictions[i] == 'O':
                        i+=1
                        continue
                    entity = predictions[i]
                    document = document_indices[i]
                    first_start, first_end = span_indices[i]
                    # Ensure that consecutive tokens with the same label are merged
                    while i < len(predictions) - 1 and predictions[i + 1] == entity:  # If inside entity, keep incrementing
                        i += 1
                    last_start, last_end = span_indices[i]
                    preds_by_document[document].append((entity, first_start, last_end))
                    i+=1

            # Write the metrics for this fold.
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

            table_data = [[label,
                           format(fold_statistics[label]['precision'], ".3f"),
                           format(fold_statistics[label]['recall'], ".3f"),
                           format(fold_statistics[label]['f1'], ".3f")]
                          for label in named_entities + ['system']]

            logging.info(tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1'],
                                  tablefmt='orgtbl'))

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

        logging.info("\n"+tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
                       tablefmt='orgtbl'))

        if prediction_directory:
            
            prediction_directory = training_dataset.get_data_directory() + "/predictions"
            groundtruth_directory = training_dataset.get_data_directory() + "/groundtruth"
            
            # Write annotations generated from cross-validation
            self.create_annotation_directory(directory=prediction_directory,training_dataset=training_dataset, option="predictions")

            # Write medaCy ground truth generated from cross-validation
            self.create_annotation_directory(directory=groundtruth_directory,training_dataset=training_dataset,option="groundtruth")
            
            #Add predicted/known annotations to the folders containing groundtruth and predictions respectively
            annotations = self.predict_annotation_evaluation(directory=groundtruth_directory,training_dataset=training_dataset, medacy_pipeline= medacy_pipeline, preds_by_document=preds_by_document, groundtruth_by_document=groundtruth_by_document, option="groundtruth")

            annotations = self.predict_annotation_evaluation(directory=prediction_directory,training_dataset=training_dataset, medacy_pipeline= medacy_pipeline, preds_by_document=preds_by_document, groundtruth_by_document=groundtruth_by_document,option="predictions")

            
            return Dataset(data_directory=prediction_directory)

    def create_annotation_directory(self, directory, training_dataset, option):
        if isinstance(directory, str):
            directory = directory
        else:
            directory = training_dataset.data_directory + "/"+option+"/"
        if os.path.isdir(directory):
            logging.warning("Overwriting existing %s",option)
        else:
            os.makedirs(directory)
        return directory

    def predict_annotation_evaluation(self, directory, training_dataset, medacy_pipeline, preds_by_document, groundtruth_by_document, option):
        for data_file in training_dataset.get_data_files():
            logging.info("Predicting %s file: %s", option, data_file.file_name)
            with open(data_file.raw_path, 'r') as raw_text:
                doc = medacy_pipeline.spacy_pipeline.make_doc(raw_text.read())
                
                if option == "groundtruth":
                    preds = groundtruth_by_document[data_file.file_name]
                else:
                    preds = preds_by_document[data_file.file_name]
                annotations = construct_annotations_from_tuples(doc, preds)
                annotations.to_ann(write_location=os.path.join(directory, data_file.file_name + ".ann"))        
        
        return annotations
    
    
    def _extract_features(self, data_file, medacy_pipeline, is_metamapped):
        """
        A multi-processed method for extracting features from a given DataFile instance.

        :param conn: pipe to pass back data to parent process
        :param data_file: an instance of DataFile
        :return: Updates queue with features for this given file.
        """
        nlp = medacy_pipeline.spacy_pipeline
        feature_extractor = medacy_pipeline.get_feature_extractor()
        logging.info("Processing file: %s", data_file.file_name)

        with open(data_file.raw_path, 'r') as raw_text:
            doc = nlp.make_doc(raw_text.read())
        # Link ann_path to doc
        doc.set_extension('gold_annotation_file', default=data_file.ann_path, force=True)
        doc.set_extension('file_name', default=data_file.file_name, force=True)

        # Link metamapped file to doc for use in MetamapComponent if exists
        if is_metamapped:
            doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

        # run 'er through
        doc = medacy_pipeline(doc)

        # The document has now been run through the pipeline. All annotations are overlayed - pull features.
        features, labels = feature_extractor(doc, data_file.file_name)

        logging.info("%s: Feature Extraction Completed (num_sequences=%i)" % (data_file.file_name, len(labels)))
        return features, labels

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

    def get_info(self, return_dict=False):
        """
        Retrieves information about a Model including details about the feature extraction pipeline, features utilized,
        and learning model.

        :param return_dict: Returns a raw dictionary of information as opposed to a formatted string
        :return: Returns structured information
        """
        pipeline_information = self.pipeline.get_pipeline_information()
        feature_extractor = self.pipeline.get_feature_extractor()
        #TODO include tokenizer
        pipeline_information['feature_extraction'] = {}
        pipeline_information['feature_extraction']['medacy_features'] = feature_extractor.all_custom_features
        pipeline_information['feature_extraction']['spacy_features'] = feature_extractor.spacy_features
        pipeline_information['feature_extraction']['window_size'] = feature_extractor.window_size

        if return_dict:
            return pipeline_information

        text = ["Pipeline Name: %s" % pipeline_information['pipeline_name'],
                "Learner Name: %s" % pipeline_information['learner_name'],
                "Pipeline Description: %s" % pipeline_information['description'],
                "Pipeline Components: [%s]" % ",".join(pipeline_information['components']),
                "Spacy Features: [%s]" % ", ".join(pipeline_information['feature_extraction']['spacy_features']),
                "Medacy Features: [%s]" % ", ".join(pipeline_information['feature_extraction']['medacy_features']).replace('feature_', ''),
                "Window Size: (+-) %i" % pipeline_information['feature_extraction']['window_size']
                ]

        return "\n".join(text)

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

    def __str__(self):
        return self.get_info()
