import shutil
import tempfile
import unittest

from medacy.data.dataset import Dataset
from medacy.model.multi_model import MultiModel
from medacy.pipelines.clinical_pipeline import ClinicalPipeline
from medacy.pipelines.testing_pipeline import TestingPipeline


class TestMultiModel(unittest.TestCase):
    """Unit tests for medacy.model.multi_model.MultiModel"""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary directory for predictions"""
        cls.temp_dir = tempfile.tempdir()

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete the temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_multi_model(self):
        """Runs all tests for valid uses of MultiModel"""

        data_1 = Dataset('TODO')
        ents_1 = data_1.get_labels()
        data_2 = Dataset('TODO')
        ents_2 = data_2.get_labels()

        multimodel = MultiModel()
        # Test that *args works
        multimodel.add_model('TODO', ClinicalPipeline, list(ents_1))
        # Test that **kwargs works
        multimodel.add_model('TODO', TestingPipeline, entities=list(ents_2))

        # Test __len__
        self.assertEqual(len(multimodel), 2)

        # Test that each model gets instantiated correctly
        for model, pipeline_class in zip(multimodel, [ClinicalPipeline, TestingPipeline]):
            current_pipeline = model.pipeline
            self.assertIsInstance(current_pipeline, pipeline_class)
            self.assertGreater(len(current_pipeline.entities), 0)

        # Test predict_directory
        resulting_data = multimodel.predict_directory(data_1.data_directory, self.temp_dir)
        labeled_items = resulting_data.get_labels()

        # Test that at least one label from each model is predicted
        self.assertTrue(any(e in ents_1 for e in labeled_items))
        self.assertTrue(any(e in ents_2 for e in labeled_items))

        # Test that all files get predicted for
        self.assertEqual(len(resulting_data), len(data_1))

    def test_errors(self):
        """Tests that invalid inputs raise the appropriate errors"""
        multimodel = MultiModel()

        # Test add_model with a nonexisting model path
        with self.assertRaises(FileNotFoundError):
            multimodel.add_model('notafilepath', ClinicalPipeline, ['Drug', 'ADE'])

        # Test add_model without passing a subclass of BasePipeline
        with self.assertRaises(TypeError):
            multimodel.add_model('TODO', 7, ['Drug', 'ADE'])

        # Test add_model without a list of entities anywhere in args or kwargs
        with self.assertRaises(ValueError):
            multimodel.add_model('TODO', TestingPipeline)
        with self.assertRaises(ValueError):
            multimodel.add_model('TODO', TestingPipeline, [])
        with self.assertRaises(ValueError):
            multimodel.add_model('TODO', TestingPipeline, entities=[6, 8])


if __name__ == '__main__':
    unittest.main()
