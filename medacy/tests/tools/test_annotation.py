import shutil,tempfile, pkg_resources
from unittest import TestCase
from medacy.data import Dataset
from medacy.tools import Annotations
from os.path import join

class TestAnnotation(TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Loads END dataset and writes files to temp directory
        :return:
        """
        cls.test_dir = tempfile.mkdtemp() #set up temp directory
        cls.dataset, cls.entities = Dataset.load_external('medacy_dataset_end')
        cls.ann_files = []
        # fill directory of training files
        for data_file in cls.dataset.get_data_files():
            file_name, raw_text, ann_text = (data_file.file_name, data_file.raw_path, data_file.ann_path)
            cls.ann_files.append(file_name + '.ann')

        with open(join(cls.test_dir, "broken_ann_file.ann"), 'w') as f:
            f.write("This is clearly not a valid ann file")


    @classmethod
    def tearDownClass(cls):
        """
        Removes test temp directory and deletes all files
        :return:
        """
        pkg_resources.cleanup_resources()
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
        annotations = Annotations(join(self.dataset.get_data_directory(),self.ann_files[0]), annotation_type='ann')
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
        Tests initialization from invalid annotation file
        :return:
        """
        with self.assertRaises(AssertionError):
            annotations = Annotations(join(self.dataset.get_data_directory(), self.ann_files[0][:-1]), annotation_type='ann')


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
            annotations = Annotations(join(self.test_dir,"broken_ann_file.ann"), annotation_type='ann')

    def test_ann_conversions(self):
        """
        Tests converting and un-converting a valid Annotations object to an ANN file.
        :return:
        """
        annotations = Annotations(join(self.dataset.get_data_directory(),self.ann_files[0]), annotation_type='ann')
        annotations.to_ann(write_location=join(self.test_dir,"intermediary.ann"))
        annotations2 = Annotations(join(self.test_dir,"intermediary.ann"), annotation_type='ann')
        self.assertEqual(annotations.get_entity_annotations(return_dictionary=True),
                          annotations2.get_entity_annotations(return_dictionary=True))

    def test_get_entity_annotations_dict(self):
        """
        Tests the validity of the annotation dict
        :return:
        """
        annotations = Annotations(join(self.dataset.get_data_directory(), self.ann_files[0]), annotation_type='ann')
        self.assertIsInstance(annotations.get_entity_annotations(return_dictionary=True), dict)

    def test_get_entity_annotations_list(self):
        """
        Tests the validity of annotation list
        :return:
        """
        annotations = Annotations(join(self.dataset.get_data_directory(), self.ann_files[0]), annotation_type='ann')
        self.assertIsInstance(annotations.get_entity_annotations(), list)





