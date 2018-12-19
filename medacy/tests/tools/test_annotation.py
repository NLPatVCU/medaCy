import os
import shutil
import tempfile
from unittest import TestCase

from medacy.data import load_END
from medacy.tools import Annotations


class TestAnnotation(TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Loads END dataset and writes files to temp directory
        :return:
        """
        cls.test_dir = tempfile.mkdtemp() #set up temp directory
        files, entities = load_END()
        ann_files = []
        for file_name, raw_text, ann_text in files:
            ann_files.append(file_name+'.ann')
            with open(os.path.join(cls.test_dir, "%s.txt" % file_name), 'w') as f:
                f.write(raw_text)
            with open(os.path.join(cls.test_dir, "%s.ann" % file_name), 'w') as f:
                f.write(ann_text)

        with open(os.path.join(cls.test_dir, "broken_ann_file.ann"), 'w') as f:
            f.write("This is clearly not a valid ann file")

        cls.entities = entities
        cls.ann_files = ann_files

    @classmethod
    def tearDownClass(cls):
        """
        Removes test temp directory and deletes all files
        :return:
        """
        shutil.rmtree(cls.test_dir) #remove temp directory and delete all files

    def test_init_from_dict(self):
        """
        Tests initalization from a dictionary
        :return:
        """
        annotations = Annotations({'entities':{}, 'relations':[]})
        self.assertIsInstance(annotations, Annotations)

    def test_init_from_ann_file(self):
        """
        Tests initialization from valid ann file
        :return:
        """
        annotations = Annotations(self.test_dir+os.path.sep+self.ann_files[0], annotation_type='ann')
        self.assertIsNotNone(annotations.get_entity_annotations())

    def test_init_from_invalid_dict(self):
        """
        Tests initialization from invalid dict file
        :return:
        """
        with self.assertRaises(AssertionError):
            annotations = Annotations({})


    def test_init_from_invalid_ann(self):
        """
        Tests initialization from invalid dict file
        :return:
        """
        with self.assertRaises(AssertionError):
            annotations = Annotations(self.test_dir+os.path.sep+self.ann_files[0][:-1], annotation_type='ann')

    def test_init_from_invalid_ann(self):
        """
        Tests initialization from invalid dict file
        :return:
        """
        with self.assertRaises(AssertionError):
            annotations = Annotations(self.test_dir+os.path.sep+self.ann_files[0][:-1], annotation_type='ann')

    def test_init_from_non_dict_or_string(self):
        """
        Tests initialization from non-dictionary or string
        :return:
        """
        with self.assertRaises(AssertionError):
            annotations = Annotations(list(), annotation_type='ann')

    def test_init_from_broken_ann_file(self):
        """
        Tests initialization from a correctly structured but ill-formated ann file
        :return:
        """
        with self.assertRaises(AssertionError):
            annotations = Annotations(self.test_dir+os.path.sep+"broken_ann_file.ann", annotation_type='ann')

    def test_ann_conversions(self):
        """
        Tests converting and un-converting a valid Annotations object to an ANN file.
        :return:
        """
        annotations = Annotations(self.test_dir + os.path.sep + self.ann_files[0], annotation_type='ann')
        annotations.to_ann(write_location=self.test_dir+os.path.sep+"intermediary.ann")
        annotations2 = Annotations(self.test_dir+os.path.sep+"intermediary.ann", annotation_type='ann')
        self.assertEqual(annotations.get_entity_annotations(return_dictionary=True),
                          annotations2.get_entity_annotations(return_dictionary=True))

    def test_get_entity_annotations_dict(self):
        """
        Tests the validity of the annotation dict
        :return:
        """
        annotations = Annotations(self.test_dir + os.path.sep + self.ann_files[0], annotation_type='ann')
        self.assertIsInstance(annotations.get_entity_annotations(return_dictionary=True), dict)

    def test_get_entity_annotations_list(self):
        """
        Tests the validity of annotation list
        :return:
        """
        annotations = Annotations(self.test_dir + os.path.sep + self.ann_files[0], annotation_type='ann')
        self.assertIsInstance(annotations.get_entity_annotations(), list)





