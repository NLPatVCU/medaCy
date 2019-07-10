"""
:author: Steele W. Farnsworth
:date: 28 May, 2019
"""

import os
import unittest
import shutil
import tempfile
from medacy.tools.converters.add_rel_to_brat import Entity, is_valid_rel, add_rel_to_brat, main
from medacy.tests.tools.converters.samples.add_rel_to_brat_samples \
    import ann_sample, rel_sample, txt_sample, expected_ann


class TestAddRelToBrat(unittest.TestCase):
    """Unit tests for add_rel_to_brat.py"""
    @classmethod
    def setUpClass(cls):
        """Create test directories and files;
        note that all files must have the same name other than the file extension"""

        cls.test_dir = tempfile.mkdtemp()

        cls.ann_txt_dir = os.path.join(cls.test_dir, "txt_ann")
        cls.rel_dir = os.path.join(cls.test_dir, "rel_dir")
        cls.all_dir = os.path.join(cls.test_dir, "all")
        cls.null_dir = os.path.join(cls.test_dir, "null")

        for f in [cls.ann_txt_dir, cls.rel_dir, cls.all_dir, cls.null_dir]:
            os.mkdir(f)

        # Create files in the correct format

        cls.test_ann_path = os.path.join(cls.ann_txt_dir, "test.ann")
        with open(cls.test_ann_path, "w+") as f:
            f.write(ann_sample)

        cls.test_rel_path = os.path.join(cls.rel_dir, "test.rel")
        with open(cls.test_rel_path, "w+") as f:
            f.write(rel_sample)

        cls.test_txt_path = os.path.join(cls.ann_txt_dir, "test.txt")
        with open(cls.test_txt_path, "w+") as f:
            f.write(txt_sample)

        # Create files in the wrong format

        cls.bad_ann_path = os.path.join(cls.all_dir, "test.ann")
        with open(cls.bad_ann_path, "w+") as f:
            f.write(ann_sample)

        cls.bad_rel_path = os.path.join(cls.all_dir, "test.rel")
        with open(cls.bad_rel_path, "w+") as f:
            f.write(rel_sample)

        cls.bad_txt_path = os.path.join(cls.all_dir, "test.txt")
        with open(cls.bad_txt_path, "w+") as f:
            f.write(txt_sample)

    @classmethod
    def tearDownClass(cls):
        """Delete test dir"""
        shutil.rmtree(cls.test_dir)

    def test_entity_eq(self):
        """Test that two equal entities are equal."""
        e1 = Entity(5, "Bob", 21, 45, "Hello world!")
        e2 = Entity(45, "Not Bob", 21, 45, "Hello world!")
        self.assertEqual(e1, e2)

    def test_entity_ne(self):
        """Test that two not equal entities are not equal"""
        e1 = Entity(5, "Bob", 56, 82, "Hello!")
        e2 = Entity(45, "Not Bob", 21, 45, "Hello world!")
        self.assertNotEqual(e1, e2)

    def test_entity_str(self):
        """Test that __str__ method is accurate"""
        e1 = Entity(5, "Bob", 21, 45, "Hello world!")
        expected = "T5\tBob 21 45\tHello world!\n"
        self.assertEqual(str(e1), expected)

    def test_is_valid_rel_true(self):
        """Test that is_valid_rel is accurate when given a valid line of rel data"""
        sample = 'c="lotions" 124:3 124:3||r="TrNAP"||c="incisions" 124:10 124:10'
        result = is_valid_rel(sample)
        self.assertTrue(result)

    def test_is_valid_rel_false(self):
        """Test that is_valid_rel is accurate when given a line that is not rel data"""
        sample = "This isn't rel"
        result = is_valid_rel(sample)
        self.assertFalse(result)

    def test_add_rel_to_brat(self):
        add_rel_to_brat(self.test_ann_path, self.test_rel_path, self.test_txt_path)
        with open(self.test_ann_path, "r") as f:
            actual = f.read()
        self.maxDiff = None
        self.assertMultiLineEqual(expected_ann, actual)

    def test_main_files_align(self):
        """Test that main runs without error when the input files are in the directories expected"""
        main([None, self.ann_txt_dir, self.rel_dir])
        self.assertTrue(True)

    def test_main_file_misalign(self):
        """Test that an error is raised when files are inputted in a different format than expected"""
        with self.assertRaises(FileNotFoundError):
            main([None, self.all_dir, self.null_dir])


if __name__ == '__main__':
    unittest.main()
