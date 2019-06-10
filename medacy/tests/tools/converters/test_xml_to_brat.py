# import unittest, tempfile, shutil
# from medacy.tools.converters.xml_to_brat import *
# from medacy.tests.tools.converters.samples.xml_samples import *
#
#
# sample_1_expected = """T1	metabolite 24 38	phosphoglucose
# T2	enzyme 24 48	phosphoglucose isomerase
# T3	metabolite 51 59	fructose
# T4	metabolite 51 76	fructose-1,6-bisphosphate
# T5	enzyme 51 85	fructose-1,6-bisphosphate aldolase
# T6	metabolite 604 618	phosphoglucose
# T7	enzyme 604 628	phosphoglucose isomerase
# """
#
# sample_2_expected = """TODO"""
#
# sample_3_expected = """TODO"""
#
#
# class TestXMLToBrat(unittest.TestCase):
#     """Unit tests for xml_to_brat.py"""
#
#     @classmethod
#     def setUpClass(cls):
#         cls.test_dir = tempfile.mkdtemp()
#
#         cls.xml_file_path_1 = os.path.join(cls.test_dir, "sample1.xml")
#         with open(cls.xml_file_path_1, "w+") as f:
#             f.write(xml_sample_1)
#
#         cls.xml_file_path_2 = os.path.join(cls.test_dir, "sample2.xml")
#         with open(cls.xml_file_path_2, "w+") as f:
#             f.write(xml_sample_2)
#
#         cls.xml_file_path_3 = os.path.join(cls.test_dir, "sample3.xml")
#         with open(cls.xml_file_path_3, "w+") as f:
#             f.write(xml_sample_3)
#
#         cls.output_file_path = os.path.join(cls.test_dir, "output_file.txt")
#
#     @classmethod
#     def tearDownClass(cls):
#         shutil.rmtree(cls.test_dir)
#
#     @unittest.skip
#     def test_xml_to_brat_1(self):
#         """Tests that extraction is accurate, even with nested entities."""
#         actual, _ = convert_xml_to_brat(self.xml_file_path_1)
#         self.maxDiff = None
#         self.assertEqual(sample_1_expected, actual)
#
#     @unittest.skip
#     def test_xml_to_brat_2(self):
#         """
#         Tests that extraction is accurate when the same entity appears many times.
#
#         Also uncovered a bug whereby the substring "&gt;" was deleted by bs4. Solution was to replace
#         that substring with ">" at certain points in the code.
#         """
#         actual, _ = convert_xml_to_brat(self.xml_file_path_2)
#         self.maxDiff = None
#         self.assertEqual(sample_2_expected, actual)
#
#     @unittest.skip
#     def test_xml_to_brat_3(self):
#         """
#         Test for a sample that had raised IndexError
#         """
#         actual, _ = convert_xml_to_brat(self.xml_file_path_3)
#         self.maxDiff = None
#         self.assertEqual(sample_3_expected, actual)
