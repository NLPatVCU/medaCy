"""
:author: Steele W. Farnsworth
:date: 6 February, 2019
"""

import unittest, tempfile, os, shutil
from medacy.tools.con_form.con_to_brat import convert_con_to_brat, is_valid_con, line_to_dict

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
c="Amphotericin B" 1:9 1:22||t="activeingredient"
c="Lipid Complex" 1:24 1:30||t="nanoparticle"
c="intravenous" 1:110 1:110||t="routeofadministration"
c="ABELCET" 2:0 2:0||t="tradename"
c="ampho-tericin B" 2:21 2:35||t="activeingredient"
c="phospholipids,L-&#x3b1;-dimyristoylphosphatidylcholine (DMPC)" 3:8 3:63||t="corecomposition"
c="L-&#x3b1;-dimyristoylphosphatidylglycerol (DMPG)," 3:74 3:116||t="corecomposition"
c="ABELCET" 4:0 4:0||t="tradename"
c="Liposomal" 5:6 5:6||t="nanoparticle"
c="lipid complex" 5:52 5:58||t="nanoparticle"
c="liposomal" 6:22 6:22||t="nanoparticle"
c="lipid-complexed" 6:35 6:35||t="nanoparticle"
c="Amphotericin B" 7:72 7:85||t="activeingredient"
c="antifungal" 7:101 7:101||t="indication"
c="Amphotericin B" 7:170 7:183||t="activeingredient"
c="924.09" 8:29 8:29||t="molecularweight"
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

    def test_is_valid_con_valid_line_1(self):
        """Tests that is_valid_con() returns True for a valid input."""
        line = "c=\"intravenous\" 1:110 1:110||t=\"routeofadministration\""
        result = is_valid_con(line)
        self.assertTrue(result)

    def test_is_valid_con_valid_line_2(self):
        """Tests that is_valid_con() returns True for a valid input."""
        line = "c=\"intra!     us\" 5:110 10:16||t=\"routeof admi nis'tration\""
        result = is_valid_con(line)
        self.assertTrue(result)

    def test_is_valid_con_invalid_line_1(self):
        """Tests that is_valid_con() returns False for an invalid input."""
        line = "c=intra!us\" 5:110 10:16||t=\"routeof admi nis'tration\""
        result = is_valid_con(line)
        self.assertFalse(result)

    def test_is_valid_con_invalid_line_2(self):
        """Tests that is_valid_con() returns False for an invalid input."""
        line = "c=intra!us\" 5:110  10:16 t=\"routeof admi nis'tration\""
        result = is_valid_con(line)
        self.assertFalse(result)

    def test_valid_brat_to_con(self):
        """Convert the test file from brat to con. Assert that the con output matches the sample con text."""
        brat_output = convert_con_to_brat(self.con_file_path, self.text_file_path)
        self.assertEqual(brat_output, brat_text)

    def test_line_to_dict_valid_line(self):
        """Tests that line_to_dict() accurately casts a line of con text to a dict."""
        line = "c=\"Amphotericin B\" 1:9 1:22||t=\"activeingredient\""
        expected = {"data_item": "Amphotericin B", "start_ind": "1:9", "end_ind": "1:22", "data_type": "activeingredient"}
        actual = line_to_dict(line)
        self.assertDictEqual(expected, actual)

    def test_invalid_file_path(self):
        """Passes an invalid file path to convert_con_to_brat()."""
        with self.assertRaises(FileNotFoundError):
            convert_con_to_brat("this isn't a valid file path", "neither is this")

    def test_valid_con_matching_text_name(self):
        """
        Assert that the con output matches the sample con text when the automatic text-file-finding feature is utilized
        """
        brat_output = convert_con_to_brat(self.con_file_path)
        self.assertEqual(brat_output, brat_text)

    def test_invalid_brat_text(self):
        """Assert that invalid brat text produces no output."""
        brat_output = convert_con_to_brat(self.bad_con_file_path, self.text_file_path)
        self.assertFalse(brat_output)
