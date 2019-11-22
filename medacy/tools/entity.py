from typing import Match

from medacy.data.data_file import DataFile
from medacy.tools.converters.brat_to_con import is_valid_brat, line_to_dict


class Entity:
    """Representation of an individual entity in an annotation document. This abstraction is not used in the Annotations
    class, but can be used to keep track of what entities are present in a document during dataset manipulation."""

    t = 1

    def __init__(self, ent_type: str, start: int, end: int, text: str, num: int = 0):
        self.num = num
        self.ent_type = ent_type
        self.start = start
        self.end = end
        self.text = text

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.text == other.text

    def __hash__(self):
        return hash((self.start, self.end, self.text))

    @classmethod
    def init_from_re_match(cls, match: Match, ent_class, num=None, increment_t=False):
        """
        Creates a new Entity from a regex Match.
        :param match: A Match object
        :param ent_class: The type of entity this is
        :param num: The number for this entity; defaults to the current entity count held by the class.
        :param increment_t: Whether or not to increment the T number
        :return: A new Entity
        """
        if not isinstance(match, Match):
            raise TypeError("Argument is not a Match object.")

        new_entity = cls(
            num=cls.t if num is None else num,
            ent_type=ent_class,
            start=match.start(),
            end=match.end(),
            text=match.string[match.start():match.end()],
        )

        if num is None and increment_t:
            # Increment the counter
            cls.t += 1

        return new_entity

    def set_t(self):
        """Sets the T value based on the class's counter and increments the counter"""
        self.num = self.__class__.t
        self.__class__.t += 1

    @classmethod
    def init_from_doc(cls, doc):
        """
        Creates a list of Entities for all entity annotations in a document.
        :param doc: can be a DataFile or str of a file path
        :return: a list of Entities
        """
        entities = []
        if isinstance(doc, DataFile):
            with open(doc.ann_path) as f:
                text_lines = f.read().split("\n")
        elif isinstance(doc, str):
            with open(doc) as f:
                text_lines = f.read().split("\n")
        for line in text_lines:
            if not is_valid_brat(line):
                continue
            line_dict = line_to_dict(line)
            new_entity = cls(
                num=line_dict["id_num"],
                ent_type=line_dict["data_type"],
                start=line_dict["start_ind"],
                end=line_dict["end_ind"],
                text=line_dict["data_item"],
            )
            entities.append(new_entity)

        cls.t = 1 + max(ent.num for ent in entities)

        return entities

    def __str__(self):
        """Returns the BRAT representation of this Entity, without a new-line character"""
        return f"T{self.num}\t{self.ent_type} {self.start} {self.end}\t{self.text}"

    def __repr__(self):
        """Return the constructor in string form"""
        return f"{type(self).__name__}({self.ent_type}, {self.start}, {self.end}, {self.text}, {self.num})"
