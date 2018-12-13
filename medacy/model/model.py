import logging, os, joblib, time
from tabulate import tabulate
from statistics import mean
from sklearn_crfsuite import metrics
from .stratified_k_fold import SequenceStratifiedKFold

from medacy.pipelines.base.base_pipeline import BasePipeline
from ..tools import DataLoader
from ..tools import model_to_ann
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count





class Model:

    def __init__(self, medacy_pipeline=None, model=None, n_jobs=cpu_count()):

        assert isinstance(medacy_pipeline, BasePipeline), "Pipeline must be a medaCy pipeline that interfaces medacy.pipelines.base.BasePipeline"

        self.pipeline = medacy_pipeline
        self.model = model

        # These arrays will store the sequences of features and sequences of corresponding labels
        self.X_data = []
        self.y_data = []
        self.n_jobs = n_jobs

        #Run an initializing document through the pipeline to register all token extensions.
        doc = self.pipeline(medacy_pipeline.spacy_pipeline.make_doc("Initialize"), predict=True)
        assert doc is not None, "Model could not be initialized with the set pipeline"

    def fit(self, training_data_loader):
        """
            Runs training data through our pipeline and fits it using the CRF algorithm
            :param training_data_loader: Instance of DataLoader containing training files
            :return model: Trained model
        """

        assert isinstance(training_data_loader, DataLoader), "Must pass in an instance of DataLoader containing your training files"
        assert isinstance(self.pipeline, BasePipeline), "Model object must contain a medacy pipeline to pre-process data"


        pool = Pool(nodes = self.n_jobs)

        results = [pool.apipe(self._extract_features, data_file, self.pipeline, training_data_loader.is_metamapped())
                   for data_file in training_data_loader.get_files()]

        while any([i.ready() == False for i in results]):
            time.sleep(1)

        for idx, i in enumerate(results):
            X,y = i.get()
            self.X_data+=X
            self.y_data+=y

        logging.info("Currently Waiting")


        learner_name, learner = self.pipeline.get_learner()
        logging.info("Training: %s", learner_name)

        assert self.X_data, "Training data is empty."

        learner.fit(self.X_data, self.y_data)
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
            doc = medacy_pipeline(doc, predict=True)

            ann_file_contents = model_to_ann(model, medacy_pipeline, doc)
            with open(prediction_directory+data_file.file_name+".ann", "a+") as f:
                f.write(ann_file_contents)


    def cross_validate(self, num_folds=10):
        """
        Performs k-fold stratified cross-validation using our model and pipeline.
        :param num_folds: number of folds to split training data into for cross validation
        :return: Prints out performance metrics
        """

        assert num_folds > 1, "Number of folds for cross validation must be greater than 1"

        assert self.model is not None, "Cannot cross validate a un-fit model"
        assert self.X_data is not None and self.y_data is not None, \
            "Must have features and labels extracted for cross validation"

        X_data = self.X_data
        Y_data = self.y_data

        medacy_pipeline = self.pipeline

        cv = SequenceStratifiedKFold(folds=num_folds)

        named_entities = medacy_pipeline.entities

        evaluation_statistics = {}
        fold = 1
        for train_indices, test_indices in cv(X_data, Y_data):
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

        logging.info(tabulate(table_data, headers=['Entity', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
                       tablefmt='orgtbl'))




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
        features, labels = feature_extractor(doc)

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

    def __str__(self):
        return self.get_info()

