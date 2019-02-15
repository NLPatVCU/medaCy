"""
Unit tests for annotations.py; contains hard-coded test data for ann files.
:author: Andriy Mulyar, Steele W. Farnsworth
:date: 12 January, 2019
"""

import shutil, tempfile, pkg_resources
from unittest import TestCase
from medacy.tools import Annotations, InvalidAnnotationError
from os.path import join, isfile
from medacy.tests.tools.con_test_data.con_test import con_text, source_text as con_source_text


ann_text_one = """T1	TimeUnits 13 25	hydroxyethyl
T2	Vehicle 25 114	)aminotris(hydroxymethyl)methane,
T3	TestArticle 183 220	Cr(V) compound
T4	Strain 247 329	a Cr(V) precursor,
T5	Vehicle 329 357	oxochromato(V)), following a
T6	TestArticle 358 390	method described elsewhere [13].
T7	GroupName 455 469	10 (per group)
T8	Vehicle 486 496	60-day-old
T9	Sex 497 501	male
T10	Strain 502 509	ICR-CD1
"""

ann_text_one_source = """None of the tests in this file currently require that the source text be accurate,
just that it exists."""

ann_text_one_modified = """T1	TimeUnits 13 25	hydroxyethyl
T2	Car 25 114	)aminotris(hydroxymethyl)methane,
T3	TestArticle 182 220	Cr(V) compound
T4	Strain 247 329	a Cr(V) precursor,
T5	Vehicle 329 357	oxochromato(V)), following a
T6	TestArticle 358 390	method described elsewhere [13].
T7	Name 455 469	10 (per group)
T8	Vehicle 486 496	60-day-old
T9	Sex 497 501	male
T10	Strain 502 509	ICR-CD1
"""

ann_text_two = """T1	Sex 372 376	Male
T2	Strain 377 387	F344/CrlBr
T3	Species 388 393	rats,
T4	Species 487 491	rats
T5	Species 768 772	rats
T6	Species 892 896	rats
T7	Species 5030 5034	rats
T8	Species 5500 5504	rats
T9	Species 5891 5895	rats
T10	Species 6012 6016	rats
"""


class TestAnnotation(TestCase):
    """Unit tests for annotations.py"""

    @classmethod
    def setUpClass(cls):
        """Creates a temp directory and places the test data strings above into files."""
        cls.test_dir = tempfile.mkdtemp()  # set up temp directory

        cls.broken_ann_file_path = join(cls.test_dir, "broken_ann_file.ann")
        with open(cls.broken_ann_file_path, 'w') as f:
            f.write("This is clearly not a valid ann file")

        cls.ann_file_path_one = join(cls.test_dir, "good_ann_file.ann")
        with open(cls.ann_file_path_one, "w+") as f:
            f.write(ann_text_one)

        cls.ann_file_one_source_path = join(cls.test_dir, "good_ann_file_source.txt")
        with open(cls.ann_file_one_source_path, "w+") as f:
            f.write(ann_text_one_source)

        cls.ann_file_path_one_modified = join(cls.test_dir, "modified_ann_file.ann")
        with open(cls.ann_file_path_one_modified, "w+") as f:
            f.write(ann_text_one_modified)

        cls.ann_file_path_two = join(cls.test_dir, "ann_file_path_two.ann")
        with open(cls.ann_file_path_two, "w+") as f:
            f.write(ann_text_two)

        # This file is created in the methods that use it; creating it in advance would cause the test
        # to always pass.
        cls.html_output_file_path = join(cls.test_dir, "html_output.html")

    @classmethod
    def tearDownClass(cls):
        """Removes test temp directory and deletes all files"""
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.test_dir)  # remove temp directory and delete all files

    def test_init_from_dict(self):
        """Tests initalization from a dictionary"""
        annotations = Annotations({'entities': {}, 'relations': []})
        self.assertIsInstance(annotations, Annotations)

    def test_init_from_ann_file(self):
        """Tests initialization from valid ann file"""
        annotations = Annotations(self.ann_file_path_one, annotation_type='ann')
        self.assertIsNotNone(annotations.get_entity_annotations())

    def test_init_from_invalid_dict(self):
        """Tests initialization from invalid dict file"""
        with self.assertRaises(InvalidAnnotationError):
            Annotations({})

    def test_init_from_invalid_ann(self):
        """Tests initialization from invalid annotation file"""
        with self.assertRaises(InvalidAnnotationError):
            Annotations(self.broken_ann_file_path, annotation_type='ann')

    def test_init_from_non_dict_or_string(self):
        """Tests initialization from non-dictionary or string"""
        with self.assertRaises(TypeError):
            Annotations(list(), annotation_type='ann')

    def test_init_from_broken_ann_path(self):
        """Tests initialization from an invalid file path"""
        with self.assertRaises(FileNotFoundError):
            Annotations("This is not a valid file path", annotation_type='ann')

    def test_ann_conversions(self):
        """Tests converting and un-converting a valid Annotations object to an ANN file."""
        annotations = Annotations(self.ann_file_path_one, annotation_type='ann')
        annotations.to_ann(write_location=join(self.test_dir,"intermediary.ann"))
        annotations2 = Annotations(join(self.test_dir, "intermediary.ann"), annotation_type='ann')
        self.assertEqual(annotations.get_entity_annotations(return_dictionary=True),
                         annotations2.get_entity_annotations(return_dictionary=True)
                         )

    def test_get_entity_annotations_dict(self):
        """Tests the validity of the annotation dict."""
        annotations = Annotations(self.ann_file_path_one, annotation_type='ann')
        self.assertIsInstance(annotations.get_entity_annotations(return_dictionary=True), dict)

    def test_get_entity_annotations_list(self):
        """Tests the validity of annotation list"""
        annotations = Annotations(self.ann_file_path_one, annotation_type='ann')
        self.assertIsInstance(annotations.get_entity_annotations(), list)

    def test_good_con_data(self):
        """Tests to see if valid con data can be used to instantiate an Annotations object."""
        with open(join(self.test_dir, "test_con.con"), 'w+') as c,\
                open(join(self.test_dir, "test_con_text.txt"), 'w+') as t:
            c.write(con_text)
            t.write(con_source_text)

            annotations = Annotations(c.name, annotation_type='con', source_text_path=t.name)
            self.assertIsInstance(annotations.get_entity_annotations(), list)

    def test_bad_con_data(self):
        """Tests to see if invalid con data will raise InvalidAnnotationError."""
        with open(join(self.test_dir, "bad_con.con"), 'w+') as c,\
                open(join(self.test_dir, "test_con_text.txt"), 'w+') as t:
            c.write("This string wishes it was a valid con file.")
            t.write("It doesn't matter what's in this file as long as it exists.")

            Annotations(c.name, annotation_type='con', source_text_path=t.name)
            self.assertRaises(InvalidAnnotationError)

    def test_good_con_data_without_text(self):
        """Tests to see if not having a source text file will raise FileNotFoundError."""
        with open(join(self.test_dir, "test_con.con"), 'w+') as c:
                c.write(con_text)
                with self.assertRaises(FileNotFoundError):
                    Annotations(c.name, annotation_type='con', source_text_path=None)

    def test_same_file_diff(self):
        """
        Tests that when a given Annotations object uses the diff() method with another Annotations object created
        from the same source file, that it returns an empty list.
        """
        annotations1 = Annotations(self.ann_file_path_one, annotation_type='ann')
        annotations2 = Annotations(self.ann_file_path_one, annotation_type='ann')
        result = annotations1.diff(annotations2)
        self.assertEqual(result, [])

    def test_different_file_diff(self):
        """
        Tests that when two different files with the same number of annotations are used in the diff() method,
        the output is a list with more than one value.
        """
        # Note that both of these files contain ten annotations
        annotations1 = Annotations(self.ann_file_path_one, annotation_type='ann')
        annotations2 = Annotations(self.ann_file_path_two, annotation_type='ann')
        result = annotations1.diff(annotations2)
        self.assertGreater(len(result), 0)

    def test_compare_by_entity_valid_data_return_dict(self):
        """Tests that when compare_by_entity() is called with valid data that it returns a dict."""
        annotations1 = Annotations(self.ann_file_path_one, annotation_type='ann')
        annotations2 = Annotations(self.ann_file_path_two, annotation_type='ann')
        result = annotations1.compare_by_entity(annotations2)
        self.assertIsInstance(result, dict)

    def test_compare_by_index_wrong_type(self):
        """
        Tests that when a valid Annotations object calls the compare_by_entity() method with an object that is not an
        Annotations, it raises ValueError.
        """
        annotations1 = Annotations(self.ann_file_path_one, annotation_type='ann')
        annotations2 = "This is not an Annotations object"
        with self.assertRaises(TypeError):
            annotations1.compare_by_index(annotations2)

    def test_compare_by_index_valid_data_return_dict(self):
        """Tests that when compare_by_index() is called with valid data, it returns a dict."""
        annotations1 = Annotations(self.ann_file_path_one, annotation_type='ann')
        annotations2 = Annotations(self.ann_file_path_two, annotation_type='ann')
        result = annotations1.compare_by_index(annotations2)
        self.assertIsInstance(result, dict)

    def test_compare_by_index_strict_lt_0(self):
        """Tests that when compare_by_index() is called with a strict below 0, it raises ValueError."""
        annotations1 = Annotations(self.ann_file_path_one, annotation_type='ann')
        annotations2 = Annotations(self.ann_file_path_two, annotation_type='ann')
        with self.assertRaises(ValueError):
            annotations1.compare_by_index(annotations2, strict=-1)

    def test_compare_by_index_all_entities_matched(self):
        """
        Tests that when compare_by_index() is called with nearly-identical data, the calculations accurately pair
        annotations that refer to the same instance of an entity but are not identical. Specifically, test that list
        "NOT_MATCHED" is empty.
        """
        annotations1 = Annotations(self.ann_file_path_one_modified)
        annotations2 = Annotations(self.ann_file_path_one)
        comparison = annotations1.compare_by_index(annotations2, strict=1)
        self.assertListEqual(comparison["NOT_MATCHED"], [])

    def test_compare_by_index_stats_equivalent_annotations_avg_accuracy_one(self):
        """
        Tests that when compare_by_index_stats() is called on two Annotations representing the same dataset, the
        average accuracy is one.
        """
        annotations1 = Annotations(self.ann_file_path_one)
        annotations2 = Annotations(self.ann_file_path_one)
        comparison = annotations1.compare_by_index_stats(annotations2)
        actual = comparison["avg_accuracy"]
        self.assertEqual(actual, 1)

    def test_compare_by_index_stats_nearly_identical_annotations_avg_accuracy(self):
        """
        Tests that when compare_by_index_stats() is called with nearly identical Annotations, the average accuracy
        is less than 1.
        """
        annotations1 = Annotations(self.ann_file_path_one_modified)
        annotations2 = Annotations(self.ann_file_path_one)
        comparison = annotations1.compare_by_index_stats(annotations2)
        accuracy = comparison["avg_accuracy"]
        self.assertLess(accuracy, 1)

    def test_stats_returns_accurate_dict(self):
        """
        Tests that when stats() is run on an annotation with three unique entities, the list
        of keys in the entity counts is three.
        """
        annotations = Annotations(self.ann_file_path_two)  # Use file two because it contains three unique entities
        stats = annotations.stats()
        num_entities = stats["entity_counts"]
        self.assertEqual(len(num_entities), 3)

    def test_to_html_no_source_text(self):
        """Tests that when to_html() is called on an annotation with no source_text_path, it throws ValueError."""
        annotations = Annotations(self.ann_file_path_one, annotation_type='ann')
        with self.assertRaises(ValueError):
            annotations.to_html(self.html_output_file_path)

    def test_to_html_with_source_text(self):
        """
        Tests that when to_html() is called on a valid object with an output file path defined, that the outputted
        HTML file is created (thus, that the method ran to completion).
        """
        assert not isfile(self.html_output_file_path), "This test requires that the html output file does not exist \
            when the test is initiated, but it already exists."
        annotations = Annotations(self.ann_file_path_one, annotation_type='ann',
                                  source_text_path=self.ann_file_one_source_path)
        annotations.to_html(self.html_output_file_path)
        self.assertTrue(isfile(self.html_output_file_path))
