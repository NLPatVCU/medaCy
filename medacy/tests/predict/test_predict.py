from unittest import TestCase
from medacy.learn import Learner
from medacy.predict import Predictor
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_components import MetaMap
import tempfile, shutil, os


class TestPredict(TestCase):

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


    def test_with_metamap(self):
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

        learner = Learner(pipeline, train_loader)

        model = learner.train()

        predictor = Predictor(pipeline, test_loader, model=model)

        predictor.predict()

        with open(predictor.prediction_directory+"predict_test.ann") as f:
            self.assertEqual(f.read(), "T1	Strength 7 11	5 mg\n")
