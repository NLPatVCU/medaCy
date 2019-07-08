import unittest
import re
from medacy.tools.entity import Entity

class TestEntity(unittest.TestCase):

    def test_init_from_re(self):
        expected = Entity("Something", 8, 18, "an example", 1)
        match = re.search("an example", "this is an example")
        actual = Entity.init_from_re_match(match, "Something")
        self.assertEqual(expected, actual)

    def test_eq(self):
        first = Entity("Something", 8, 18, "an example", 1)
        second = Entity("Else", 8, 18, "an example", 5)
        self.assertEqual(first, second)

