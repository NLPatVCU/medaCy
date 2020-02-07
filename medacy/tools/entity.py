import os
from typing import Match

from medacy.data.data_file import DataFile
from medacy.data.annotations import Annotations


class Entity:
    """
    Representation of an individual entity in an annotation document. This abstraction is not used in the Annotations
    class, but can be used to keep track of what entities are present in a document during dataset manipulation.

    :ivar tag: the tag of this Entity
    :ivar start: the start index
    :ivar end: the end index
    :ivar text: the text of the Entity
    """

    t = 1

    def __init__(self, tag: str, start: int, end: int, text: str, num: int = 0):
        self.num = num
        self.tag = tag
        self.start = start
        self.end = end
        self.text = text

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.tag == other.tag

    def __hash__(self):
        return hash((self.start, self.end, self.text))

    def __str__(self):
        """Returns the BRAT representation of this Entity, without a new-line character"""
        return f"T{self.num}\t{self.tag} {self.start} {self.end}\t{self.text}"

    def __repr__(self):
        """Return the constructor in string form"""
        return f"{type(self).__name__}({self.tag}, {self.start}, {self.end}, {self.text}, {self.num})"

    @classmethod
    def reset_t(cls):
        """
        Resest the T counter for this class to 1
        :return: The previous value of t
        """
        previous = cls.t
        cls.t = 1
        return previous

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
            tag=ent_class,
            start=match.start(),
            end=match.end(),
            text=match.string[match.start():match.end()],
        )

        if num is None and increment_t:
            # Increment the counter
            cls.t += 1

        return new_entity

    @classmethod
    def init_from_doc(cls, doc):
        """
        Creates a list of Entities for all entity annotations in a document.
        :param doc: can be a DataFile or str of a file path
        :return: a list of Entities
        """
        if isinstance(doc, DataFile):
            ann = Annotations(doc.ann_path, doc.txt_path)
        elif isinstance(doc, (str, os.PathLike)):
            ann = Annotations(doc)
        else:
            raise ValueError(f"'doc'' must be DataFile, str, or os.PathLike, but is '{type(doc)}'")

        entities = []

        for ent in ann:
            # Entities are a tuple of (label, start, end, text)
            new_ent = cls(
                tag=ent[0],
                start=ent[1],
                end=ent[2],
                text=ent[3]
            )
            entities.append(new_ent)

        return entities

    def set_t(self):
        """Sets the T value based on the class's counter and increments the counter"""
        self.num = self.__class__.t
        self.__class__.t += 1

    def equals(self, other, mode='strict'):
        """
        Determines if two Entities match, based on if the spans match and the tag is the same.
        If mode is set to 'lenient', two Entities match if the other span is fully within or fully
        without the first Entity and the tag is the same.
        :param other: another instance of Entity
        :param mode: 'strict' or 'lenient'; defaults to 'strict'
        :return: True or False
        """
        if not isinstance(other, Entity):
            raise ValueError(f"'other' must be another instance of Entity, but is '{type(other)}'")

        if mode == 'strict':
            return self == other
        if mode != 'lenient':
            raise ValueError(f"'mode' must be 'strict' or 'lenient', but is '{mode}'")

        # Lenient
        return ((self.end > other.start and self.start < other.end) or (self.start < other.end and other.start < self.end)) and self.tag == other.tag


def sort_entities(entities):
    """
    Sorts a list of Entity instances, adjusting the num value of each one
    :param entities: a list of Entities
    :return: a sorted list; all instances have ascending num values starting at 1
    """
    if not all(isinstance(e, Entity) for e in entities):
        raise ValueError("At least one item in entities is not an Entity")

    entities = entities.copy()
    entities.sort(key=lambda x: (x.start, x.end))

    for i, e in enumerate(entities, 1):
        e.num = i

    return entities
