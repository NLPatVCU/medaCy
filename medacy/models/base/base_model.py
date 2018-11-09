from abc import ABC, abstractmethod
from medacy.pipelines.base.base_pipeline import BasePipeline
import joblib, os

class BaseModel(ABC):
    """
        A base model class defining the implementation of how the model fits, evaluates, and predicts on medical data
    """

    def __init__(self, name, medacy_pipeline=None, model=None):

        assert isinstance(medacy_pipeline, BasePipeline), "Pipeline must be a medaCy pipeline that interfaces medacy.pipelines.base.BasePipeline"

        self.name = name
        self.pipeline = medacy_pipeline
        self.model = model


    def get_model_name(self):
        return self.name


    def get_pipeline_name(self):
        return self.pipeline.pipeline_name


    def set_pipeline(self, medacy_pipeline):
        assert isinstance(medacy_pipeline, BasePipeline), "Pipeline must be a medaCy pipeline that interfaces medacy.pipelines.base.BasePipeline"
        self.pipeline = medacy_pipeline


    @abstractmethod
    def fit(self, training_data_loader):
        """
            Runs training data the MedaCy pipeline and fits the model to the pre-procssed data
            :param training_data_loader: DataLoader instance containing the files to be used for training
            :return:
        """
        pass


    @abstractmethod
    def predict(self, new_data_loader):
        """
            Uses model to make predictions on new data
            Prediction appear in a /predictions sub-directory of your data.
            :param new_data_loader: DataLoader instance containing new examples for prediction
            :return:
        """
        pass


    @abstractmethod
    def cross_validate(self, training_data_loader, num_folds=10):
        """
            Uses model to make predictions on new data
            Prediction appear in a /predictions sub-directory of your data.
            :param training_data_loader: DataLoader instance containing training data
            :param num_folds: int for how many folds to use for cross-validation
            :return:
        """
        pass


    def dump(self, destination):
        """
        Pickles and dumps the model to a specified directory
        :param destination: Directory to dump model into
        :return:
        """
        with open(os.path.join(destination, 'model.txt'), 'wb') as f:
            model = {
                'pipeline': self.pipeline,
                'model': self.model
            }
            joblib.dump(model,f)
        return


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Model Name: ' + self.get_model_name() + '\n' + 'Pipeline Name: ' + self.pipeline.pipeline_name