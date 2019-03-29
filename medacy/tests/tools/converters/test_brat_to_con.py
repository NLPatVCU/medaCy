"""
:author: Steele W. Farnsworth
:date: 13 March, 2019
"""

import unittest, tempfile
from medacy.tools.converters.brat_to_con import *

brat_text = """T1	tradename 0 7	ABELCET
T2	activeingredient 9 23	Amphotericin B
T3	nanoparticle 24 37	Lipid Complex
T4	routeofadministration 110 121	intravenous
T5	tradename 132 139	ABELCET
T6	activeingredient 153 168	ampho-tericin B
T7	corecomposition 246 307	phospholipids,L-&#x3b1;-dimyristoylphosphatidylcholine (DMPC)
T8	corecomposition 312 361	L-&#x3b1;-dimyristoylphosphatidylglycerol (DMPG),
T9	tradename 397 404	ABELCET
T10	nanoparticle 468 477	Liposomal
T11	nanoparticle 514 527	lipid complex
T12	nanoparticle 674 683	liposomal
T13	nanoparticle 687 702	lipid-complexed
T14	activeingredient 911 925	Amphotericin B
T15	indication 940 950	antifungal
T16	activeingredient 1009 1023	Amphotericin B
T17	molecularweight 1397 1403	924.09
T18	tradename 1470 1477	ABELCET
"""

con_text = """c="ABELCET" 1:0 1:0||t="tradename"
c="Amphotericin B" 1:1 1:2||t="activeingredient"
c="Lipid Complex" 1:3 1:4||t="nanoparticle"
c="intravenous" 1:12 1:12||t="routeofadministration"
c="ABELCET" 2:0 2:0||t="tradename"
c="ampho-tericin B" 2:3 2:4||t="activeingredient"
c="phospholipids,L-&#x3b1;-dimyristoylphosphatidylcholine (DMPC)" 3:2 3:3||t="corecomposition"
c="L-&#x3b1;-dimyristoylphosphatidylglycerol (DMPG)," 3:5 3:6||t="corecomposition"
c="ABELCET" 4:0 4:0||t="tradename"
c="Liposomal" 5:1 5:1||t="nanoparticle"
c="lipid complex" 5:7 5:8||t="nanoparticle"
c="liposomal" 6:2 6:2||t="nanoparticle"
c="lipid-complexed" 6:4 6:4||t="nanoparticle"
c="Amphotericin B" 7:8 7:9||t="activeingredient"
c="antifungal" 7:13 7:13||t="indication"
c="Amphotericin B" 7:21 7:22||t="activeingredient"
c="924.09" 8:6 8:6||t="molecularweight"
c="ABELCET" 10:0 10:0||t="tradename"
"""

source_text = """ABELCET (Amphotericin B Lipid Complex Injection)DESCRIPTIONABELCET  is a sterile, pyrogen-free suspension for intravenous infusion.
ABELCET  consists of ampho-tericin B complexed with two phospholipids in a 1:1 drug-to-lipid molar ratio.
The two phospholipids,L-&#x3b1;-dimyristoylphosphatidylcholine (DMPC) and L-&#x3b1;-dimyristoylphosphatidylglycerol (DMPG), are pre-sent in a 7:3 molar ratio.
ABELCET  is yellow and opaque in appearance, with a pH of 5 - 7.
NOTE: Liposomal encapsulation or incorporation in a lipid complex can substantially affect adrug's functional properties relative to those of the unencapsulated or nonlipid-associated drug.
Inaddition, different liposomal or lipid-complexed products with a common active ingredient mayvary from one another in the chemical composition and physical form of the lipid component.
Suchdifferences may affect functional properties of these drug products.Amphotericin B is a polyene, antifungal antibiotic produced from a strain of Streptomyces nodosus.Amphotericin B is designated chemically as [1R-(1R*, 3S*, 5R*, 6R*, 9R*, 11R*, 15S*, 16R*, 17R*,18S*, 19E, 21E, 23E, 25E, 27E, 29E, 31E, 33R*, 35S*, 36R*, 37S*)]-33-[(3-Amino-3, 6- D-mannopyranosyl) oxy]-1,3,5,6,9,11,17,37-octahydroxy-15,16,18-trimethyl-13-oxo-14,39-dioxabicy-clo[33.3.1] nonatriaconta-19, 21, 23, 25, 27, 29, 31-heptaene-36-carboxylic acid.
It has a molecular weight of 924.09 and a molecular formula of C47H73NO17.
The structural formula is:
ABELCET  is provided as a sterile, opaque suspension in 20 mL glass, single-use vials.
"""

bad_brat_text = """Animadvertit qui mihi proximus recumbebat, et an probarem interrogavit.
Negavi. 'Tu ergo' inquit 'quam consuetudinem sequeris?'
'Eadem omnibus pono; ad cenam enim, non ad notam
invito cunctisque rebus exaequo, quos mensa et toro aequavi.'
"""


class TestBratToCon(unittest.TestCase):
    """Unit tests for brat_to_con.py"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        cls.brat_file_path = os.path.join(cls.test_dir, "good_brat_file.ann")
        with open(cls.brat_file_path, "w+") as f:
            f.write(brat_text)

        cls.con_file_path = os.path.join(cls.test_dir, "good_con_file.con")
        with open(cls.con_file_path, "w+") as f:
            f.write(con_text)

        # The name of this text file must match cls.brat_file_path (minus the extension) for
        # test_valid_brat_matching_text_name() to pass.
        cls.text_file_path = os.path.join(cls.test_dir, "good_brat_file.txt")
        with open(cls.text_file_path, "w+") as f:
            f.write(source_text)

        cls.bad_brat_file_path = os.path.join(cls.test_dir, "bad_brat_file.ann")
        with open(cls.bad_brat_file_path, "w+") as f:
            f.write(bad_brat_text)

        cls.output_file_path = os.path.join(cls.test_dir, "output_file.txt")

        cls.lines = Line.init_lines(source_text)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_is_valid_brat_valid_1(self):
        """Tests that when is_valid_brat() gets called on a valid line without a new line character, it returns True."""
        sample = "T3	nanoparticle 24 37	Lipid Complex"
        result = is_valid_brat(sample)
        self.assertTrue(result)

    def test_is_valid_brat_valid_2(self):
        """Tests that when is_valid_brat() is called on a valid line with a new line character, it returns True."""
        sample = "T12	nanoparticle 674 683	liposomal\n"
        result = is_valid_brat(sample)
        self.assertTrue(result)

    def test_is_valid_brat_invalid_1(self):
        """Tests what when is_valid_brat() is called on an invalid line without a new line character, it returns False."""
        sample = "T3	nanoparticle s 37	Lipid Complex"
        result = is_valid_brat(sample)
        self.assertFalse(result)

    def test_is_valid_brat_invalid_2(self):
        """Tests what when is_valid_brat() is called on an invalid line with a new line character, it returns False."""
        sample = "T12 674 683	liposomal\n"
        result = is_valid_brat(sample)
        self.assertFalse(result)

    def test_line_to_dict(self):
        """Tests that line_to_dict() accurately converts a line of input text to an expected dict format."""
        sample = "T3	nanoparticle 24 37	Lipid Complex"
        expected = {"id_type": "T", "id_num": 3, "data_type": "nanoparticle", "start_ind": 24,
                    "end_ind": 37, "data_item": "Lipid Complex"}
        actual = line_to_dict(sample)
        self.assertDictEqual(expected, actual)

    def test_switch_extension(self):
        """Tests that switch_extension() accurately switches the file extension."""
        sample = "some_file.rar"
        expected = "some_file.txt"
        actual = switch_extension(sample, ".txt")
        self.assertEqual(expected, actual)

    def test_get_word_num_1(self):
        """
        Test that get_word_num() accurately identifies the word index for a given instance
        of a word in a line; the word only occurs once in its line.
        """
        # The annotation used is "T5	tradename 132 139	ABELCET"
        sample_line = "ABELCET  consists of ampho-tericin B complexed with two phospholipids in a 1:1 drug-to-lipid molar ratio."
        this_line = self.lines[1]
        expected = 0
        actual = get_word_num(this_line, 132)
        self.assertEqual(actual, expected)

    def test_get_word_num_2(self):
        """
        Test that get_word_num() accurately identifies the word index for a given instance
        of a word in a line when the word appears more than once in that line
        """
        # The annotation used is "T16	activeingredient 1009 1023	Amphotericin B"
        sample_line = "Suchdifferences may affect functional properties of these drug products.Amphotericin B is a polyene, antifungal antibiotic produced from a strain of Streptomyces nodosus.Amphotericin B is designated chemically as [1R-(1R*, 3S*, 5R*, 6R*, 9R*, 11R*, 15S*, 16R*, 17R*,18S*, 19E, 21E, 23E, 25E, 27E, 29E, 31E, 33R*, 35S*, 36R*, 37S*)]-33-[(3-Amino-3, 6- D-mannopyranosyl) oxy]-1,3,5,6,9,11,17,37-octahydroxy-15,16,18-trimethyl-13-oxo-14,39-dioxabicy-clo[33.3.1] nonatriaconta-19, 21, 23, 25, 27, 29, 31-heptaene-36-carboxylic acid."
        this_line = self.lines[6]
        expected = 21
        actual = get_word_num(this_line, 1009)
        self.assertEqual(expected, actual)

    def test_valid_brat_to_con(self):
        """Convert the test file from brat to con. Assert that the con output matches the sample con text."""
        con_output = convert_brat_to_con(self.brat_file_path, self.text_file_path)
        self.assertEqual(con_text, con_output)

    def test_invalid_file_path(self):
        """Passes an invalid file path to convert_brat_to_con()."""
        with self.assertRaises(FileNotFoundError):
            convert_brat_to_con("this isn't a valid file path", "neither is this")
            
    def test_valid_brat_matching_text_name(self):
        """
        Assert that the con output matches the sample con text when the automatic text-file-finding feature is utilized
        """
        con_output = convert_brat_to_con(self.brat_file_path)
        self.assertEqual(con_text, con_output)

    def test_invalid_brat_text(self):
        """Assert that invalid brat text produces no output."""
        con_output = convert_brat_to_con(self.bad_brat_file_path, self.text_file_path)
        self.assertFalse(con_output)
