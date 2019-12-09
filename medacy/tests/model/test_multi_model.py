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
        """Runs all tests for MultiModel"""

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

        # Test that labels from both models get predicted for
        self.assertSetEqual(ents_1 | ents_2, labeled_items)

        # Test that all files get predicted for
        self.assertEqual(len(resulting_data), len(data_1))


if __name__ == '__main__':
    unittest.main()
