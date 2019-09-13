import pkg_resources
import shutil
import tempfile
from os.path import join
from unittest import TestCase, skip

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset


class TestAnnotation(TestCase):

    @classmethod
    def setUpClass(cls):
        """Loads END dataset and writes files to temp directory"""
        cls.test_dir = tempfile.mkdtemp()  # set up temp directory
        cls.good_dataset = Dataset("./sample_data/")

        with open(join(cls.test_dir, "broken_ann_file.ann"), 'w') as f:
            f.write("This is clearly not a valid ann file")

        cls.ann_1 = Annotations(cls.good_dataset.data_files[0].ann_path)
        cls.ann_2 = Annotations(cls.good_dataset.data_files[1].ann_path)

    @classmethod
    def tearDownClass(cls):
        """Removes test temp directory and deletes all files"""
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.test_dir)

    def test_init_from_ann_file(self):
        """Tests initialization from valid ann file"""
        self.assertIsNotNone(self.ann_1.annotations)

    def test_init_from_invalid_ann(self):
        """Tests initialization from invalid annotation file"""
        with self.assertRaises(FileNotFoundError):
            Annotations("not_a_file_path")

    def test_ann_conversions(self):
        """Tests converting and un-converting a valid Annotations object to an ANN file."""
        self.maxDiff = None
        annotations = self.ann_1
        annotations.to_ann(write_location=join(self.test_dir, "intermediary.ann"))
        annotations2 = Annotations(join(self.test_dir, "intermediary.ann"))
        self.assertListEqual(annotations.annotations, annotations2.annotations)

    def test_difference(self):
        """Tests that when a given Annotations object uses the diff() method with another Annotations object created
        from the same source file, that it returns an empty list."""
        result = self.ann_1.difference(self.ann_1)
        self.assertFalse(result)

    def test_different_file_diff(self):
        """Tests that when two different files are used in the difference method, the output is a list with more than
        one value."""
        result = self.ann_1.difference(self.ann_2)
        self.assertGreater(len(result), 0)

    def test_compute_ambiguity(self):
        label, start, end, text = self.ann_1.annotations[0]
        ann_2_copy = Annotations(self.good_dataset.data_files[1].ann_path)
        ann_2_copy.add_entity('incorrect_label', start, end, text)
        self.assertEqual(len(self.ann_1.compute_ambiguity(ann_2_copy)), 1)

    @skip("Not currently working")
    def test_confusion_matrix(self):
        annotations1 = Annotations(self.good_dataset.data_files[1].ann_path)
        annotations2 = self.ann_2
        annotations1.add_entity(*annotations2.get_entity_annotations()[0])
        self.assertEqual(len(annotations1.compute_confusion_matrix(annotations2, self.entities)[0]), len(self.entities))
        self.assertEqual(len(annotations1.compute_confusion_matrix(annotations2, self.entities)), len(self.entities))

    @skip("Not currently working")
    def test_intersection(self):
        annotations1 = Annotations(self.good_dataset.data_files[0].ann_path)
        annotations2 = Annotations(self.good_dataset.data_files[1].ann_path)
        annotations1.add_entity(*annotations2.annotations[0])
        annotations1.add_entity(*annotations2.annotations[1])
        expected = {annotations2.annotations[0], annotations2.annotations[1]}
        self.assertSetEqual(annotations1.intersection(annotations2), expected)

    def test_compute_counts(self):
        self.assertIsInstance(self.ann_1.compute_counts(), dict)
