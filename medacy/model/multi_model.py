import os
from shutil import copyfile

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines.base.base_pipeline import BasePipeline


def _activate_model(model_path, pipeline_class, args, kwargs):
    """
    Creates a Model with the given pipeline configuration and sets its weights to the pickled model path
    :param model_path: path to the model pickle file
    :param pipeline_class: the pipeline class for the pickled model
    :param args, kwargs: arguments to pass to the pipeline constructor
    :return: a usable Model instance
    """
    pipeline_instance = pipeline_class(*args, **kwargs)
    model = Model(pipeline_instance)
    model.load(model_path)
    return model


class MultiModel:
    """
    Allows for prediction with multiple models, ensuring that only the model being used at a given time and its pipeline
    are present in memory.

    An example use case:
    >>> from medacy.model.multi_model import MultiModel
    >>> from medacy.pipelines.clinical_pipeline import ClinicalPipeline
    >>> from medacy.pipelines.scispacy_pipeline import ScispacyPipeline
    >>> multimodel = MultiModel()
    >>> multimodel.add_model('path/to/model_one.pkl', ClinicalPipeline, ['Drug', 'ADE'])
    >>> multimodel.add_model('path/to/model_two.pkl', ScispacyPipeline, ['Dose', 'Frequency'])
    >>> for model in multimodel:
    ...     model.predict('The patient was prescribed 5mg of Tylenol and got a headache.')
    >>> predicted_data = multimodel.predict_directory('path/to/input/data', 'path/to/output/directory')
    """

    def __init__(self):
        """No values are needed to instantiate a new MultiModel."""
        self.models = []

    def __len__(self):
        return len(self.models)

    def add_model(self, model_path, pipeline_class, *args, **kwargs):
        """
        Adds a new model to the MultiModel
        :param model_path: path to the model pickle file
        :param pipeline_class: the pipeline class for the pickled model
        :param args, kwargs: arguments to pass to the pipeline constructor
        :return: None
        """

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"'model path' is not a path to an existing file, but is {repr(model_path)}")
        if not issubclass(pipeline_class, BasePipeline):
            raise TypeError(f"'pipeline_class' must be a subclass of BasePipeline, but is '{repr(pipeline_class)}'")

        self.models.append((model_path, pipeline_class, args, kwargs))

    def __iter__(self):
        """
        Individually activates and returns usable Model instances one at a time
        """
        for tup in self.models:
            model_path, pipeline, args, kwargs = tup
            yield _activate_model(model_path, pipeline, args, kwargs)

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
        annotation_dict = {f: Annotations([], source_text_path=f) for f in txt_files}

        for model in self:
            for file_name in txt_files:
                file_path = os.path.join(data_directory, file_name)
                with open(file_path) as f:
                    text = f.read()
                this_annotations = annotation_dict[file_name]
                resulting_annotations = model.predict(text)
                # Merge the two Annotations together and store them back in the dictionary
                annotation_dict[file_name] = this_annotations | resulting_annotations

        # Create the new Dataset directory
        for path, ann in annotation_dict.items():
            # Get the name of the output ann file
            path = os.path.join(data_directory, path)
            base_name = os.path.basename(path)[:-4]
            output_ann = os.path.join(prediction_directory, base_name + '.ann')
            output_txt = os.path.join(prediction_directory, base_name + '.txt')

            # Write the ann file
            ann.to_ann(output_ann)
            # Copy the txt file
            copyfile(path, output_txt)

        return Dataset(prediction_directory)
