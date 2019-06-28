import unittest
import os
from medacy.tools.evaluator import evaluate_annotation_agreement

class TestEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Store the locations of the sample data as attributes."""
        location = os.path.abspath(os.path.dirname(__file__))
        cls.gold_data_dir_name = os.path.join(location, "sample_data/gold_ann_samples")
        cls.false_data_dir_name = os.path.join(location, "sample_data/false_ann_samples")

    def test_evaluator(self):
        actual = evaluate_annotation_agreement(
            self.gold_data_dir_name,
            self.gold_data_dir_name,
            entities=["adversereaction", "tradename"],
            relations=["Drug-ADE"],
            verbose=False
        )
        print(actual)
