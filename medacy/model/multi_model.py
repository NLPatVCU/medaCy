import logging
import os
from shutil import copyfile

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.model._model import predict_document
from medacy.model.model import Model
from medacy.pipelines.base.base_pipeline import BasePipeline


def _activate_model(pipeline, model_path):
    """
    Creates a Model with the given pipeline and sets its weights to the pickled model path
    :param pipeline: the pipeline instance to be used for the pickled model
    :param model_path: path to the model pickle file
    :return: a usable Model instance
    """
    model = Model(pipeline)
    model.load(model_path)
    return model


class MultiModel:
    """Allows for prediction with multiple models, ensuring that only the model being used at a given time is
    present in memory."""

    def __init__(self):
        """No values are needed to instantiate a new MultiModel; You can set self.models to a list of pipeline and
        pickle file path tuples, but this bypasses the validation performed by self.add_model()"""
        self.models = []
        self._labels = set()

    def __len__(self):
        return len(self.models)

    def add_model(self, pipeline, model_path):
        """
        Adds a new model to the MultiModel
        :param pipeline: the pipeline instance to be used for the pickled model
        :param model_path: path to the model pickle file
        :return: None
        """

        if not isinstance(pipeline, BasePipeline):
            raise TypeError(f"'pipeline' must be instance of a subclass of BasePipeline, but is '{repr(pipeline)}'")
        if not os.path.isfile(model_path):
            raise FileNotFoundError("'model path' is not a path to an existing file")

        # Warn user is incoming pipeline predicts for entities that another model predicts for
        new_labels = set(pipeline.entities)
        labels_already_present = self._labels & new_labels
        if labels_already_present:
            logging.warning(f"These labels are being predicted for by more than one model: {labels_already_present}")
        self._labels.update(new_labels)

        self.models.append((pipeline, model_path))

    def generate_models(self):
        """
        Individually activates and returns usable Model instances one at a time
        """
        for pipeline, model_path in self.models:
            yield _activate_model(pipeline, model_path)

    def __iter__(self):
        raise NotImplementedError("Please use generate_models() when iterating over a MultiModel")

    def predict_directory(self, data_directory, prediction_directory):
        """
        Predicts over all txt files in a directory using every Model. Note that this method spends a lot of time
        on file IO because each txt file is opened as many times as there are models.
        :param data_directory: Path to a directory of text files to predict over
        :param prediction_directory: a directory to write predictions to
        :return: a Dataset of the predictions
        """
        if not os.path.isdir(data_directory):
            raise ValueError(f"'data_directory' must be an existing directory, but is '{repr(data_directory)}'")
        if not os.path.isdir(prediction_directory):
            raise ValueError(f"'prediction_directory' must be a directory, but is '{repr(prediction_directory)}'")

        # Get all the txt files in the input directory
        txt_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
        # Create a dictionary of empty Annotations objects to store the predictions
        annotation_dict = {f: Annotations([]) for f in txt_files}

        for model in self.generate_models():
            pipeline = model.pipeline
            for file_name in txt_files:
                with open(file_name) as f:
                    text = f.read()
                doc = pipeline(text)
                this_annotations = annotation_dict[file_name]
                resulting_annotations = predict_document(model, doc, pipeline)
                # Merge the two Annotations together and store them back in the dictionary
                annotation_dict[file_name] = this_annotations | resulting_annotations

        # Create the new Dataset directory
        for path, ann in annotation_dict.items():
            # Get the name of the output ann file
            base_name = os.path.basename(path)
            output_ann = os.path.join(prediction_directory, base_name + '.ann')
            output_txt = os.path.join(prediction_directory, base_name + '.txt')

            # Write the ann file
            ann.to_ann(output_ann)
            # Copy the txt file
            copyfile(path, output_txt)

        return Dataset(prediction_directory)
