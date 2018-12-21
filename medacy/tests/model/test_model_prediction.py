from unittest import TestCase
from medacy.model import Model
from medacy.pipelines import ClinicalPipeline
from medacy.tools import DataLoader, Annotations
from medacy.pipeline_components import MetaMap
from medacy.data import load_END
import tempfile, shutil, os


class TestModelBulk(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_dir = tempfile.mkdtemp() #set up train directory
        cls.test_dir = tempfile.mkdtemp()  # set up predict directory
        files, entities = load_END()
        cls.entities = entities
        cls.ann_files = []

        #fill directory of training files
        for file_name, raw_text, ann_text in files:
            cls.ann_files.append(file_name + '.ann')
            with open(os.path.join(cls.train_dir, "%s.txt" % file_name), 'w') as f:
                f.write(raw_text)
            with open(os.path.join(cls.train_dir, "%s.ann" % file_name), 'w') as f:
                f.write(ann_text)

            #place same files into evalutation directory.
            with open(os.path.join(cls.test_dir, "%s.txt" % file_name), 'w') as f:
                f.write(raw_text)
            with open(os.path.join(cls.test_dir, "%s.ann" % file_name), 'w') as f:
                f.write(ann_text)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.train_dir)
        shutil.rmtree(cls.test_dir)


    def test_prediction_with_clinical_pipeline(self):
        """
        Constructs a model that memorizes an entity, predicts it on same file, writes to ann
        :return:
        """

        train_loader = DataLoader(self.train_dir, limit=1)
        test_loader = DataLoader(self.test_dir, limit=1)
        metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap",
                          cache_output=False)

        pipeline = ClinicalPipeline(metamap, entities=['tradename'])

        model = Model(pipeline, n_jobs=1)
        model.fit(train_loader)

        model.predict(test_loader)
        annotations = Annotations(os.path.join(self.test_dir, "predictions",self.ann_files[0]), annotation_type='ann')

        self.assertIsInstance(annotations, Annotations)
