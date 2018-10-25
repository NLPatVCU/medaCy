from unittest import TestCase
from medacy.learn import Learner
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_components import MetaMap
import tempfile, shutil, os


class TestLearn(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp() #set up test directory
        with open(os.path.join(cls.test_dir, 'test.txt'), 'w') as f:
            f.write("I took 5 mg")

        with open(os.path.join(cls.test_dir, 'test.ann'), 'w') as f:
            f.write("T1	Strength 8 11	5 mg")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)


    def test_with_metamap(self):

        loader = DataLoader(self.test_dir)
        metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap")

        loader.metamap(metamap) #pre-cache metamap files

        pipeline = ClinicalPipeline(metamap, entities=['Strength'])


        learner = Learner(pipeline, loader)

        model = learner.train()

        self.assertIsInstance(learner, Learner)
