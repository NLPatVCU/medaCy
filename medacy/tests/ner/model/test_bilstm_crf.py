from unittest import TestCase
from medacy.ner.model import Model
from medacy.ner.pipelines import LstmSystematicReviewPipeline
from medacy.tools import Annotations
from medacy.data import Dataset
import os, importlib, pkg_resources, tempfile, shutil


class TestBiLstmCrf(TestCase):
    """
    Tests model training and prediction in bulk
    """

    @classmethod
    def setUpClass(cls):

        if importlib.util.find_spec('medacy_dataset_end') is None:
            raise ImportError("medacy_dataset_end was not automatically installed for testing. See testing instructions for details.")

        cls.train_dataset, _,  meta_data = Dataset.load_external('medacy_dataset_end')
        cls.entities = meta_data['entities']
        cls.train_dataset.set_data_limit(1)

        cls.test_dataset, _, _ = Dataset.load_external('medacy_dataset_end')
        cls.test_dataset.set_data_limit(2)

        cls.prediction_directory = tempfile.mkdtemp() #directory to store predictions



    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)


    def test_prediction_with_testing_pipeline(self):
        """
        Constructs a model that memorizes an entity, predicts it on same file, writes to ann
        :return:
        """

        pipeline = LstmSystematicReviewPipeline(
            entities=['tradename'],
            word_embeddings='medacy/tests/ner/model/test_word_embeddings.txt',
            cuda_device=-1
        )

        #train on Abelcet.ann
        model = Model(pipeline)
        model.fit(self.train_dataset)

        #predict on both
        model.predict(self.test_dataset, prediction_directory=self.prediction_directory)

        second_ann_file = "%s.ann" % self.test_dataset.get_data_files()[1].file_name
        annotations = Annotations(os.path.join(self.prediction_directory, second_ann_file), annotation_type='ann')
        self.assertIsInstance(annotations, Annotations)
