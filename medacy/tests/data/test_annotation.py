import shutil
import tempfile
from os.path import join
from unittest import TestCase

import pkg_resources

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset


class TestAnnotation(TestCase):

    @classmethod
    def setUpClass(cls):
        """Loads END dataset and writes files to temp directory"""
        cls.test_dir = tempfile.mkdtemp()  # set up temp directory
        cls.good_dataset, _ = Dataset.load_external('medacy_dataset_end')
        cls.entities = list(cls.good_dataset.get_labels())

        with open(join(cls.test_dir, "broken_ann_file.ann"), 'w') as f:
            f.write("This is clearly not a valid ann file")

        cls.ann_path_1 = cls.good_dataset.all_data_files[0].ann_path
        cls.ann_path_2 = cls.good_dataset.all_data_files[1].ann_path

    @classmethod
    def tearDownClass(cls):
        """Removes test temp directory and deletes all files"""
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.test_dir)

    def test_init_from_ann_file(self):
        """Tests initialization from valid ann file"""
        ann = Annotations(self.ann_path_1)
        self.assertIsNotNone(ann.annotations)

    def test_init_from_invalid_ann(self):
        """Tests initialization from invalid annotation file"""
        with self.assertRaises(FileNotFoundError):
            Annotations("not_a_file_path")

    def test_ann_conversions(self):
        """Tests converting and un-converting a valid Annotations object to an ANN file."""
        self.maxDiff = None
        annotations = Annotations(self.ann_path_1)
        annotations.to_ann(write_location=join(self.test_dir, "intermediary.ann"))
        annotations2 = Annotations(join(self.test_dir, "intermediary.ann"))
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
        label, start, end, text = ann_1.annotations[0]
        ann_1_copy = Annotations(self.ann_path_1)
        ann_1_copy.add_entity('incorrect_label', start, end, text)
        ambiguity = ann_1.compute_ambiguity(ann_1_copy)
        self.assertEqual(len(ambiguity), 1)

    def test_confusion_matrix(self):
        ann_1 = Annotations(self.ann_path_1)
        ann_2 = Annotations(self.ann_path_2)
        ann_1.add_entity(*ann_2.get_entity_annotations()[0])
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
