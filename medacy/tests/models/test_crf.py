from unittest import TestCase
from medacy.models import CRFModel
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader
from medacy.pipeline_components import MetaMap

import tempfile, shutil, os, joblib

class TestCRF(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_dir = tempfile.mkdtemp()  # set up train directory
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

    def test_with_clinical_pipeline(self):
        train_loader = DataLoader(self.train_dir)
        test_loader = DataLoader(self.test_dir)
        metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap",
                          cache_output=False)

        train_loader.metamap(metamap)
        test_loader.metamap(metamap)

        pipeline = ClinicalPipeline(metamap, entities=['Strength'])
        model = CRFModel(pipeline)
        assert model.get_model_name() == 'CRF', "Expected model name to be 'CRF', not " + model.get_model_name()
        assert model.get_pipeline_name() == 'clinical_pipeline', "Expected pipeline name to be 'ClinicalPipeline', not " + model.get_pipeline_name()
        model.fit(train_loader)
        model.predict(test_loader)
        with open(test_loader.data_directory + "/predictions/" + "predict_test.ann") as f:
            self.assertEqual(f.read(), "T1	Strength 7 11	5 mg\n")

    def test_dumping(self):
        metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap",
                          cache_output=False)
        pipeline = ClinicalPipeline(metamap, entities=['Strength'])
        model = CRFModel(pipeline)
        os.makedirs(self.test_dir + '/dump')
        model.dump(self.test_dir + '/dump/')
        with open(self.test_dir + "/dump/model.txt", "rb") as f:
            obj = joblib.load(f)
            assert obj['model'] == None and isinstance(obj['pipeline'], ClinicalPipeline), "Model did not properly dump, dumping result was: " + obj.__str__()


