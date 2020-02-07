import unittest

from medacy.data.dataset import Dataset
from medacy.tests.sample_data import sample_dataset
from medacy.tools.calculators.inter_dataset_agreement import Measures, measure_dataset


class TestDatasetAgreement(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gold_dataset = sample_dataset
        cls.predicted_dataset = Dataset(str(sample_dataset.data_directory) + "_predictions")
        cls.maxDiff = None

    def test_agreement(self):
        strict_expected = {
            'CellLine': Measures(tp=7, fp=0, tn=0, fn=0),
            'Dose': Measures(tp=29, fp=3, tn=0, fn=2),
            'DoseDuration': Measures(tp=5, fp=0, tn=0, fn=0),
            'DoseDurationUnits': Measures(tp=5, fp=0, tn=0, fn=0),
            'DoseFrequency': Measures(tp=2, fp=0, tn=0, fn=0),
            'DoseRoute': Measures(tp=5, fp=26, tn=0, fn=10),
            'DoseUnits': Measures(tp=23, fp=3, tn=0, fn=3),
            'Endpoint': Measures(tp=33, fp=151, tn=0, fn=78),
            'EndpointUnitOfMeasure': Measures(tp=17, fp=16, tn=0, fn=11),
            'GroupName': Measures(tp=11, fp=5, tn=0, fn=4),
            'GroupSize': Measures(tp=11, fp=0, tn=0, fn=0),
            'SampleSize': Measures(tp=3, fp=0, tn=0, fn=0),
            'Sex': Measures(tp=17, fp=0, tn=0, fn=1),
            'Species': Measures(tp=41, fp=1, tn=0, fn=6),
            'Strain': Measures(tp=6, fp=0, tn=0, fn=0),
            'TestArticle': Measures(tp=33, fp=87, tn=0, fn=32),
            'TestArticlePurity': Measures(tp=1, fp=0, tn=0, fn=0),
            'TimeAtFirstDose': Measures(tp=0, fp=2, tn=0, fn=1),
            'TimeEndpointAssessed': Measures(tp=8, fp=2, tn=0, fn=2),
            'TimeUnits': Measures(tp=8, fp=1, tn=0, fn=1),
            'Vehicle': Measures(tp=13, fp=5, tn=0, fn=4),
            'system': Measures(tp=278, fp=302, tn=0, fn=155)
        }

        lenient_expected = {
             'CellLine': Measures(tp=7, fp=0, tn=0, fn=0),
             'Dose': Measures(tp=30, fp=1, tn=0, fn=1),
             'DoseDuration': Measures(tp=5, fp=0, tn=0, fn=0),
             'DoseDurationUnits': Measures(tp=5, fp=0, tn=0, fn=0),
             'DoseFrequency': Measures(tp=2, fp=0, tn=0, fn=0),
             'DoseRoute': Measures(tp=15, fp=5, tn=0, fn=0),
             'DoseUnits': Measures(tp=26, fp=0, tn=0, fn=0),
             'Endpoint': Measures(tp=99, fp=27, tn=0, fn=12),
             'EndpointUnitOfMeasure': Measures(tp=26, fp=3, tn=0, fn=2),
             'GroupName': Measures(tp=14, fp=0, tn=0, fn=1),
             'GroupSize': Measures(tp=11, fp=0, tn=0, fn=0),
             'SampleSize': Measures(tp=3, fp=0, tn=0, fn=0),
             'Sex': Measures(tp=17, fp=0, tn=0, fn=1),
             'Species': Measures(tp=42, fp=0, tn=0, fn=5),
             'Strain': Measures(tp=6, fp=0, tn=0, fn=0),
             'TestArticle': Measures(tp=63, fp=30, tn=0, fn=2),
             'TestArticlePurity': Measures(tp=1, fp=0, tn=0, fn=0),
             'TimeAtFirstDose': Measures(tp=1, fp=0, tn=0, fn=0),
             'TimeEndpointAssessed': Measures(tp=9, fp=0, tn=0, fn=1),
             'TimeUnits': Measures(tp=9, fp=0, tn=0, fn=0),
             'Vehicle': Measures(tp=16, fp=0, tn=0, fn=1),
             'system': Measures(tp=407, fp=66, tn=0, fn=26)
        }

        entities = self.gold_dataset.get_labels()

        # Test strict
        strict_actual = measure_dataset(self.gold_dataset, self.predicted_dataset, mode='strict')
        for ent in entities:
            expected = strict_expected[ent]
            actual = strict_actual[ent]
            self.assertEqual(actual, expected)

        # Test lenient
        lenient_actual = measure_dataset(self.gold_dataset, self.predicted_dataset, mode='lenient')
        for ent in entities:
            expected = lenient_expected[ent]
            actual = lenient_actual[ent]
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
