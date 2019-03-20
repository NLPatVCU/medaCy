import unittest
from medacy.tools.converters.conversion_tools.line import Line


# Sample text must be on the lowest level of indentation so that
# the indentation is not counted towards the indices.

sample_text_1 = """ABELCET (Amphotericin B Lipid Complex Injection)DESCRIPTIONABELCET  is a sterile, pyrogen-free suspension for intravenous infusion.
ABELCET  consists of ampho-tericin B complexed with two phospholipids in a 1:1 drug-to-lipid molar ratio.
The two phospholipids,L-&#x3b1;-dimyristoylphosphatidylcholine (DMPC) and L-&#x3b1;-dimyristoylphosphatidylglycerol (DMPG), are pre-sent in a 7:3 molar ratio.
ABELCET  is yellow and opaque in appearance, with a pH of 5 - 7.
NOTE: Liposomal encapsulation or incorporation in a lipid complex can substantially affect adrug's functional properties relative to those of the unencapsulated or nonlipid-associated drug.
Inaddition, different liposomal or lipid-complexed products with a common active ingredient mayvary from one another in the chemical composition and physical form of the lipid component.
Suchdifferences may affect functional properties of these drug products.Amphotericin B is a polyene, antifungal antibiotic produced from a strain of Streptomyces nodosus.Amphotericin B is designated chemically as [1R-(1R*, 3S*, 5R*, 6R*, 9R*, 11R*, 15S*, 16R*, 17R*,18S*, 19E, 21E, 23E, 25E, 27E, 29E, 31E, 33R*, 35S*, 36R*, 37S*)]-33-[(3-Amino-3, 6- D-mannopyranosyl) oxy]-1,3,5,6,9,11,17,37-octahydroxy-15,16,18-trimethyl-13-oxo-14,39-dioxabicy-clo[33.3.1] nonatriaconta-19, 21, 23, 25, 27, 29, 31-heptaene-36-carboxylic acid.
It has a molecular weight of 924.09 and a molecular formula of C47H73NO17.
The structural formula is:
ABELCET  is provided as a sterile, opaque suspension in 20 mL glass, single-use vials."""

sample_text_2 = """This is the first sample line
This is the second line
Also this line
This is another line
Also this line
The previous line is a repeat on purpose
Also this line
This is so much fun"""


class TestLine(unittest.TestCase):
    """Unit tests for line.py"""

    def test_init_lines_no_repeats(self):
        """Test that indices are accurate when there are no repeated lines."""
        text_lines = sample_text_1.split('\n')
        line_objs = Line.init_lines(sample_text_1)
        expected = [sample_text_1.index(line) for line in text_lines]
        actual = [line.index for line in line_objs]
        self.assertListEqual(actual, expected)

    def test_init_lines_with_repeats(self):
        """Test that indices are accurate even when lines are repeated."""
        line_objs = Line.init_lines(sample_text_2)
        expected = [0, 30, 54, 69, 90, 105, 146, 161]
        actual = [line.index for line in line_objs]
        self.assertListEqual(actual, expected)
