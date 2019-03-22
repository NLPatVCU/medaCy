"""
:author: Steele W. Farnsworth
:date: 13 March, 2019
"""

import unittest, tempfile
from medacy.tools.converters.con_to_brat import *

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

bad_con_text = """'Etiamne libertos?' 'Etiam; convictores enim tunc, non libertos puto.'
Et ille: 'Magno tibi constat.' 'Minime.'
'Qui fieri potest?' 'Quia scilicet liberti mei non idem quod ego bibunt,
sed idem ego quod liberti.'
"""


class TestConToBrat(unittest.TestCase):
    """Unit tests for con_to_brat.py"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        cls.brat_file_path = os.path.join(cls.test_dir, "good_brat_file.ann")
        with open(cls.brat_file_path, "w+") as f:
            f.write(brat_text)

        cls.con_file_path = os.path.join(cls.test_dir, "good_con_file.con")
        with open(cls.con_file_path, "w+") as f:
            f.write(con_text)

        # The name of this text file must match cls.con_file_path (minus the extension) for
        # test_valid_con_matching_text_name() to pass.
        cls.text_file_path = os.path.join(cls.test_dir, "good_con_file.txt")
        with open(cls.text_file_path, "w+") as f:
            f.write(source_text)

        cls.bad_con_file_path = os.path.join(cls.test_dir, "bad_con_file.con")
        with open(cls.bad_con_file_path, "w+") as f:
            f.write(bad_con_text)

        cls.output_file_path = os.path.join(cls.test_dir, "output_file.txt")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_is_valid_con_valid_1(self):
        """Test that is_valid_con() returns True for valid text without a new-line character."""
        sample = "c=\"lipid complex\" 5:7 5:8||t=\"nanoparticle\""
        result = is_valid_con(sample)
        self.assertTrue(result)

    def test_is_valid_con_valid_2(self):
        """Test that is_valid_con() returns True for valid text with a new line character."""
        sample = "c=\"antif\"ungal\" 7:13 7:13||t=\"indica\"tion\"\n"
        result = is_valid_con(sample)
        self.assertTrue(result)

    def test_is_valid_con_invalid_1(self):
        """Test that is_valid_con() returns False for invalid text."""
        sample = "c=\"lipid complex\" 5:n 5:8||t=\"nanoparticle\""
        result = is_valid_con(sample)
        self.assertFalse(result)

    def test_is_valid_con_invalid_2(self):
        """Test that is_valid_con() returns False for invalid text."""
        sample = "c=\"antifungal 7:13 7:13||t=\"indication\"\n"
        result = is_valid_con(sample)
        self.assertFalse(result)

    def test_line_to_dict(self):
        """Test that line_to_dict() accurately converts a line of con text to a dict."""
        sample = "c=\"Amphotericin B\" 7:8 7:9||t=\"activeingredient\""
        expected = {"data_item": "Amphotericin B", "start_ind": "7:8", "end_ind": "7:9", "data_type": "activeingredient"}
        actual = line_to_dict(sample)
        self.assertDictEqual(expected, actual)

    @unittest.skip
    def test_valid_brat_to_con(self):
        """Convert the test file from brat to con. Assert that the con output matches the sample con text."""
        brat_output = convert_con_to_brat(self.con_file_path, self.text_file_path)
        self.assertEqual(brat_text, brat_output)

    def test_invalid_file_path(self):
        """Passes an invalid file path to convert_con_to_brat()."""
        with self.assertRaises(FileNotFoundError):
            convert_con_to_brat("this isn't a valid file path", "neither is this")

    @unittest.skip
    def test_valid_con_matching_text_name(self):
        """
        Assert that the con output matches the sample con text when the automatic text-file-finding feature is utilized
        """
        brat_output = convert_con_to_brat(self.con_file_path)
        self.assertEqual(brat_text, brat_output)

    def test_invalid_brat_text(self):
        """Assert that invalid brat text produces no output."""
        brat_output = convert_con_to_brat(self.bad_con_file_path, self.text_file_path)
        self.assertFalse(brat_output)
