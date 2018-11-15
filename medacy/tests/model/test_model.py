from unittest import TestCase
from medacy.model import Model
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_components import MetaMap
import tempfile, shutil, os


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_dir = tempfile.mkdtemp() #set up train directory
        cls.test_dir = tempfile.mkdtemp()  # set up predict directory
        with open(os.path.join(cls.train_dir, 'train_test.txt'), 'w') as f:
            f.write("I took 5 mg")

        with open(os.path.join(cls.train_dir, 'train_test.ann'), 'w') as f:
            f.write("T1	Strength 8 11	5 mg")

        with open(os.path.join(cls.test_dir, 'predict_test.txt'), 'w') as f:
            f.write("I took 5 mg")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.train_dir)
        shutil.rmtree(cls.test_dir)

    def test_fit_with_clinical_pipeline(self):
        """
        Loads in training data and uses it to fit a model using the Clinical Pipeline
        :return:
        """
        train_loader = DataLoader(self.train_dir)
        metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap",
                          cache_output=False)

        train_loader.metamap(metamap)

        pipeline = ClinicalPipeline(metamap, entities=['Strength'])

        model = Model(pipeline)
        model.fit(train_loader)

        self.assertIsInstance(model, Model)
        self.assertIsNot(model.model, None)

    def test_prediction_with_clinical_pipeline(self):
        """
        Constructs a model that memorizes an entity, predicts it on same file, writes to ann
        :return:
        """

        train_loader = DataLoader(self.train_dir)
        test_loader = DataLoader(self.test_dir)
        metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap",
                          cache_output=False)

        train_loader.metamap(metamap)
        test_loader.metamap(metamap)

        pipeline = ClinicalPipeline(metamap, entities=['Strength'])

        model = Model(pipeline)
        model.fit(train_loader)
        model.predict(test_loader)

        with open(self.test_dir + "/predictions/" + "predict_test.ann") as f:
            self.assertEqual(f.read(), "T1	Strength 7 11	5 mg\n")
