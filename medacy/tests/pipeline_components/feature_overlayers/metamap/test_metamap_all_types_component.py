import re
import unittest

from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_all_types_component import MetaMapAllTypesComponent
from medacy.pipelines.testing_pipeline import TestingPipeline
from medacy.tools.get_metamap import get_metamap


class TestMetaMapAllTypesComponent(unittest.TestCase):
    """Test cases for MetaMapAllTypesComponent"""

    @classmethod
    def setUpClass(cls) -> None:
        """Instantiates MetaMap and the pipeline for these tests"""
        cls.metamap = MetaMap(get_metamap())

        class TestPipeline(TestingPipeline):
            def __init__(self, ents):
                super().__init__(entities=ents)
                self.add_component(MetaMapAllTypesComponent, cls.metamap)

        cls.Pipeline = TestPipeline

    def test_create_metamap(self):
        """Tests creating a MetaMap object from path in the config file"""
        self.metamap = MetaMap(get_metamap())
        self.assertIsInstance(self.metamap, MetaMap)

    def test_has_all_types(self):
        """
        Tests that all tokens are given every semantic type label found in the dataset; this ensures that
        when a new semantic type is found, old Docs retroactively receive the corresponding annotation
        """
        data = Dataset("/home/steele/projects/medaCy/medacy/tests/sample_data/sample_dataset")
        ents = data.get_labels(as_list=True)
        pipe = self.Pipeline(ents=ents)
        model = Model(pipe)

        all_sem_types = set()
        for mm_file in [d.metamapped_path for d in data]:
            with open(mm_file) as f:
                text = f.read()
            all_sem_types |= set(re.findall("(?<=\"SemType\": \")[a-z]+(?=\")", text))

        # Strictly speaking, this test is only looking at the target words rather than all words in the window
        type_labels = ['0:feature_is_' + sem_type for sem_type in all_sem_types]

        model.preprocess(data)

        for entry in model.X_data:
            for entry_dict in entry[0]:
                for label in type_labels:
                    self.assertIn(label, entry_dict.keys())


if __name__ == '__main__':
    unittest.main()
