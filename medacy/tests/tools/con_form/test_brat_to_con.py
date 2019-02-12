"""
:author: Steele W. Farnsworth
:date: 28 December, 2018
"""

import unittest, tempfile, os, shutil
from medacy.tools.con_form.brat_to_con import convert_brat_to_con

brat_text = """T1	tradename 0 7	ABELCET
T2	activeingredient 9 23	Amphotericin B
T3	nanoparticle 24 37	Lipid Complex
T4	tradename 66 66	
T5	routeofadministration 110 121	intravenous
T6	tradename 132 139	ABELCET
T7	activeingredient 153 168	ampho-tericin B
T8	corecomposition 246 307	phospholipids,L-&#x3b1;-dimyristoylphosphatidylcholine (DMPC)
T9	corecomposition 312 361	L-&#x3b1;-dimyristoylphosphatidylglycerol (DMPG),
T10	tradename 397 404	ABELCET
T11	nanoparticle 468 477	Liposomal
T12	nanoparticle 514 527	lipid complex
T13	nanoparticle 674 683	liposomal
T14	nanoparticle 687 702	lipid-complexed
T15	activeingredient 911 925	Amphotericin B
T16	indication 940 950	antifungal
T17	activeingredient 1009 1023	Amphotericin B
T18	molecularweight 1397 1403	924.09
T19	tradename 1470 1477	ABELCET
"""

con_text = """c="ABELCET" 1:0 1:0||t="tradename"
c="Amphotericin B" 1:9 1:22||t="activeingredient"
c="Lipid Complex" 1:24 1:30||t="nanoparticle"
c="" 1:66 1:66||t="tradename"
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

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_valid_brat_to_con(self):
        """Convert the test file from brat to con. Assert that the con output matches the sample con text."""
        con_output = convert_brat_to_con(self.brat_file_path, self.text_file_path)
        self.assertEqual(con_output, con_text)

    def test_invalid_file_path(self):
        """Passes an invalid file path to convert_brat_to_con()."""
        with self.assertRaises(FileNotFoundError):
            convert_brat_to_con("this isn't a valid file path", "neither is this")

    def test_valid_brat_matching_text_name(self):
        """
        Assert that the con output matches the sample con text when the automatic text-file-finding feature is utilized
        """
        con_output = convert_brat_to_con(self.brat_file_path)
        self.assertEqual(con_output, con_text)

    def test_invalid_brat_text(self):
        """Assert that invalid brat text produces no output."""
        con_output = convert_brat_to_con(self.bad_brat_file_path, self.text_file_path)
        self.assertFalse(con_output)


