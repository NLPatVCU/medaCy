import os
import shutil
import tempfile
import unittest

import pkg_resources

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.tests.sample_data import test_dir


class TestAnnotation(unittest.TestCase):
    """Tests for medacy.data.annotations.Annotations"""

    @classmethod
    def setUpClass(cls):
        """Loads sample dataset and sets up a temporary directory for IO tests"""
        cls.test_dir = tempfile.mkdtemp()  # set up temp directory
        cls.sample_data_dir = os.path.join(test_dir, 'sample_dataset_1')
        cls.dataset = Dataset(cls.sample_data_dir)
        cls.entities = cls.dataset.get_labels(as_list=True)

        with open(os.path.join(cls.test_dir, "broken_ann_file.ann"), 'w') as f:
            f.write("This is clearly not a valid ann file")

        cls.ann_path_1 = cls.dataset.data_files[0].ann_path
        cls.ann_path_2 = cls.dataset.data_files[1].ann_path

    @classmethod
    def tearDownClass(cls):
        """Removes test temp directory and deletes all files"""
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.test_dir)

    def _test_is_sorted(self, ann):
        expected = sorted([e[1] for e in ann])
        actual = [e[1] for e in ann]
        self.assertListEqual(actual, expected)

    def test_init_from_ann_file(self):
        """Tests initialization from valid ann file"""
        ann = Annotations(self.ann_path_1)
        self._test_is_sorted(ann)

    def test_init_from_invalid_ann(self):
        """Tests initialization from invalid annotation file"""
        with self.assertRaises(FileNotFoundError):
            Annotations("not_a_file_path")

    def test_init_tuples(self):
        """Tests the creation of individual annotation tuples, including ones with non-contiguous spans"""
        temp_path = os.path.join(self.test_dir, 'tuples.ann')

        samples = [
            ("T1\tObject 66 77\tthis is some text\n", ('Object', 66, 77, 'this is some text')),
            ("T2\tEntity 44 55;66 77\tI love NER\n", ('Entity', 44, 77, 'I love NER')),
            ("T3\tThingy 66 77;88 99;100 188\tthis is some sample text\n", ('Thingy', 66, 188, 'this is some sample text'))
        ]

        for string, expected in samples:
            with open(temp_path, 'w') as f:
                f.write(string)

            resulting_ann = Annotations(temp_path)
            actual = resulting_ann.annotations[0]
            self.assertTupleEqual(actual, expected)

    def test_ann_conversions(self):
        """Tests converting and un-converting a valid Annotations object to an ANN file."""
        self.maxDiff = None
        annotations = Annotations(self.ann_path_1)
        temp_path = os.path.join(self.test_dir, "intermediary.ann")
        annotations.to_ann(write_location=temp_path)
        annotations2 = Annotations(temp_path)
        self.assertListEqual(annotations.annotations, annotations2.annotations)

    def test_difference(self):
        """Tests that when a given Annotations object uses the diff() method with another Annotations object created
        from the same source file, that it returns an empty list."""
        ann = Annotations(self.ann_path_1)
        result = ann.difference(ann)
        self.assertFalse(result)

    def test_different_file_diff(self):
        """Tests that when two different files are used in the difference method, the output is a list with more than
        one value."""
        ann_1 = Annotations(self.ann_path_1)
        ann_2 = Annotations(self.ann_path_2)
        result = ann_1.difference(ann_2)
        self.assertGreater(len(result), 0)

    def test_compute_ambiguity(self):
        ann_1 = Annotations(self.ann_path_1)
        ann_1_copy = Annotations(self.ann_path_1)
        ambiguity = ann_1.compute_ambiguity(ann_1_copy)
        # The number of overlapping spans for the selected ann file is known to be 25
        self.assertEqual(25, len(ambiguity))
        # Manually introduce ambiguity by changing the name of an entity in the copy
        first_tuple = ann_1_copy.annotations[0]
        ann_1_copy.annotations[0] = ('different_name', first_tuple[1], first_tuple[2], first_tuple[3])
        ambiguity = ann_1.compute_ambiguity(ann_1_copy)
        # See if this increased the ambiguity score by one
        self.assertEqual(26, len(ambiguity))

    def test_confusion_matrix(self):
        ann_1 = Annotations(self.ann_path_1)
        ann_2 = Annotations(self.ann_path_2)
        ann_1.add_entity(*ann_2.annotations[0])
        self.assertEqual(len(ann_1.compute_confusion_matrix(ann_2, self.entities)[0]), len(self.entities))
        self.assertEqual(len(ann_1.compute_confusion_matrix(ann_2, self.entities)), len(self.entities))

    def test_intersection(self):
        ann_1 = Annotations(self.ann_path_1)
        ann_2 = Annotations(self.ann_path_2)
        ann_1.add_entity(*ann_2.annotations[0])
        ann_1.add_entity(*ann_2.annotations[1])
        expected = {ann_2.annotations[0], ann_2.annotations[1]}
        actual = ann_1.intersection(ann_2)
        self.assertSetEqual(actual, expected)

    def test_compute_counts(self):
        ann_1 = Annotations(self.ann_path_1)
        self.assertIsInstance(ann_1.compute_counts(), dict)

    def test_or(self):
        """
        Tests that the pipe operator correctly merges two Annotations and retains the source text path of
        the left operand
        """
        tup_1 = ('Object', 66, 77, 'this is some text')
        tup_2 = ('Entity', 44, 77, 'I love NER')
        tup_3 = ('Thingy', 66, 188, 'this is some sample text')
        file_name = 'some_file'

        ann_1 = Annotations([tup_1, tup_2], source_text_path=file_name)
        ann_2 = Annotations([tup_3])

        for a in [ann_1, ann_2]:
            self._test_is_sorted(a)

        # Test __or__
        result = ann_1 | ann_2
        expected = {tup_1, tup_2, tup_3}
        actual = set(result)
        self.assertSetEqual(actual, expected)
        self.assertEqual(file_name, result.source_text_path)
        self._test_is_sorted(result)

        # Test __ior__
        ann_1 |= ann_2
        actual = set(ann_1)
        self.assertSetEqual(actual, expected)
        self._test_is_sorted(ann_1)


if __name__ == '__main__':
    unittest.main()
