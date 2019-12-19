import os
import shutil
import tempfile
import unittest

from medacy.data.dataset import Dataset
from medacy.model.multi_model import MultiModel
from medacy.pipelines.clinical_pipeline import ClinicalPipeline
from medacy.pipelines.testing_pipeline import TestingPipeline
from medacy.tests.sample_data import test_dir


class TestMultiModel(unittest.TestCase):
    """Unit tests for medacy.model.multi_model.MultiModel"""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary directory for predictions"""
        cls.temp_dir = tempfile.mkdtemp()

        cls.data_dir = os.path.join(test_dir, 'sample_dataset_1')

        cls.sample_model_1_path = os.path.join(test_dir, 'sample_models', 'sample_clin_pipe.pkl')
        cls.sample_model_2_path = os.path.join(test_dir, 'sample_models', 'sample_test_pipe.pkl')

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete the temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_multi_model(self):
        """Runs all tests for valid uses of MultiModel"""

        data = Dataset(self.data_dir)
        ents_1 = {'Endpoints', 'Species', 'DoseUnits'}
        ents_2 = {'TestArticle', 'Dose', 'Sex'}

        multimodel = MultiModel()
        # Test that *args works
        multimodel.add_model(self.sample_model_1_path, ClinicalPipeline, list(ents_1))
        # Test that **kwargs works
        multimodel.add_model(self.sample_model_2_path, TestingPipeline, entities=list(ents_2))

        # Test __len__
        self.assertEqual(len(multimodel), 2)

        # Test that each model gets instantiated correctly
        for model, pipeline_class in zip(multimodel, [ClinicalPipeline, TestingPipeline]):
            current_pipeline = model.pipeline
            self.assertIsInstance(current_pipeline, pipeline_class)
            self.assertGreater(len(current_pipeline.entities), 0)

        # Test predict_directory
        resulting_data = multimodel.predict_directory(data.data_directory, self.temp_dir)
        labeled_items = resulting_data.get_labels()

        # Test that at least one label from each model is predicted
        self.assertTrue(any(e in ents_1 for e in labeled_items))
        self.assertTrue(any(e in ents_2 for e in labeled_items))

        # Test that all files get predicted for
        self.assertEqual(len(resulting_data), len(data))

    def test_errors(self):
        """Tests that invalid inputs raise the appropriate errors"""
        multimodel = MultiModel()

        # Test add_model with a nonexisting model path
        with self.assertRaises(FileNotFoundError):
            multimodel.add_model('notafilepath', ClinicalPipeline)

        # Test add_model without passing a subclass of BasePipeline
        with self.assertRaises(TypeError):
            multimodel.add_model(self.sample_model_1_path, 7)


if __name__ == '__main__':
    unittest.main()
