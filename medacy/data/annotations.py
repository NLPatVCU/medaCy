import logging
import os
import re
from collections import Counter
from math import ceil


def is_valid_brat(item: str):
    """Returns a boolean value for whether or not a given line is a BRAT entity."""
    return isinstance(item, str) and re.fullmatch(r"T\d+\t\S+ \d+ \d+\t.+", item, re.DOTALL)


class Annotations:
    """
    An Annotations object stores all relevant information needed to manage Annotations over a document.

    The Annotation object is utilized by medaCy to structure input to models and output from models.
    This object wraps a list of tuples representing the entities in a document.
    """

    def __init__(self, annotation_data, source_text_path=None):
        """
        :param annotation_data: a file path to an annotation file
        :param source_text_path: path to the text file from which the annotations were derived; optional for ann files but necessary for conversion to or from con.
        """
        if isinstance(annotation_data, list):
            self.annotations = annotation_data
            return
        elif not os.path.isfile(annotation_data):
            raise ValueError("annotation_data must be a list or a valid file path, but is %s" % repr(annotation_data))

        self.ann_path = annotation_data
        self.source_text_path = source_text_path
        self.annotations = []

        with open(annotation_data, 'r') as f:
            annotation_lines = f.read().split("\n")

        for letter in ['R', 'E', 'A', 'M', 'N']:
            if letter in [line[:1] for line in annotation_lines]:
                logging.warning("Note that only entity annotations are implemented in medaCy")
                break

        for line in annotation_lines:
            line = line.strip()
            if not is_valid_brat(line): continue
            line = line.split("\t")

            tags = line[1].split(" ")
            entity_name = tags[0]
            entity_start = int(tags[1])
            entity_end = int(tags[-1])
            text = line[-1]
            self.annotations.append((entity_name, entity_start, entity_end, text))

    def get_labels(self):
        """
        Get the set of labels from this collection of annotations.
        :return: The list of labels.
        """
        labels = set(e[0] for e in self.annotations)
        return list(labels)

    def get_entity_annotations(self, format='medacy', nlp=None):
        """
        Returns a list of entity annotation tuples

        :return: A list of entities or underlying dictionary of entities
        """
        if format == 'medacy':
            return self.annotations
        elif format == 'spacy':
            if not self.source_text_path:
                raise FileNotFoundError("spaCy format requires the source text path")

            with open(self.source_text_path, 'r') as source_text_file:
                source_text = source_text_file.read()

            entities = []

            for tag, start, end, text in self.annotations:
                entities.append((start, end, tag))

            return source_text, entities
        else:
            raise ValueError("'%s' is not a valid annotation format" % format)

    def add_entity(self, label, start, end, text=""):
        """
        Adds an entity to the Annotations

        :param label: the label of the annotation you are appending
        :param start: the start index in the document of the annotation you are appending.
        :param end: the end index in the document of the annotation you are appending
        :param text: the raw text of the annotation you are appending
        """
        self.annotations.append((label, start, end, text))

    def to_ann(self, write_location=None):
        """
        Formats the Annotations object into a string representing a valid ANN file. Optionally writes the formatted
        string to a destination.

        :param write_location: path of location to write ann file to
        :return: returns string formatted as an ann file, if write_location is valid path also writes to that path.
        """
        ann_string = ""
        for num, tup in enumerate(self.annotations, 1):
            entity, first_start, last_end, labeled_text = tup
            ann_string += "%s\t%s %i %i\t%s\n" % (num, entity, first_start, last_end, labeled_text.replace('\n', ' '))

        if write_location is not None:
            if os.path.isfile(write_location):
                logging.warning("Overwriting file at: %s", write_location)
            with open(write_location, 'w') as file:
                file.write(ann_string)

        return ann_string

    def difference(self, other, leniency=0):
        """
        Identifies the difference between two Annotations objects. Useful for checking if an unverified annotation
        matches an annotation known to be accurate. This is done returning a list of all annotations in the operated on
        Annotation object that do not exist in the passed in annotation object. This is a set difference.

        :param other: Another Annotations object.
        :param leniency: a floating point value between [0,1] defining the leniency of the character spans to count
        as different. A value of zero considers only exact character matches while a positive value considers entities
        that differ by up to :code:`ceil(leniency * len(span)/2)` on either side.
        :return: A set of tuples of non-matching annotations.
        """
        if not isinstance(other, Annotations):
            raise ValueError("Annotations.diff() can only accept another Annotations object as an argument.")
        if leniency == 0:
            return set(self.annotations) - set(other.annotations)
        if not  0 <= leniency <= 1:
            raise ValueError("Leniency must be a floating point between [0,1]")

        matches = set()
        for label, start, end, text in self.annotations:
            window = ceil(leniency * (end - start))
            for c_label, c_start, c_end, c_text in other.annotations:
                if label == c_label:
                    if start - window <= c_start and end + window >= c_end:
                        matches.add((label, start, end, text))
                        break

        return set(self.annotations) - matches

    def intersection(self, other, leniency=0):
        """
        Computes the intersection of the operated annotation object with the operand annotation object.

        :param other: Another Annotations object.
        :param leniency: a floating point value between [0,1] defining the leniency of the character spans to count as
        different. A value of zero considers only exact character matches while a positive value considers entities that
         differ by up to :code:`ceil(leniency * len(span)/2)` on either side.
        :return A set of annotations that appear in both Annotation objects
        """
        if not isinstance(other, Annotations):
            raise ValueError("An Annotations object is requried as an argument.")
        if leniency != 0:
            if not  0 <= leniency <= 1:
                raise ValueError("Leniency must be a floating point between [0,1]")

        matches = set()
        for label, start, end, text in self.annotations:
            window = ceil(leniency * (end - start))
            for c_label, c_start, c_end, c_text in other.get_entity_annotations():
                if label == c_label and start - window <= c_start and end + window >= c_end:
                    matches.add((label, start, end, text))
                    break

        return matches

    def compute_ambiguity(self, other):
        """
        Finds occurrences of spans from 'annotations' that intersect with a span from this annotation but do not have this spans label.
        label. If 'annotation' comprises a models predictions, this method provides a strong indicators
        of a model's in-ability to dis-ambiguate between entities. For a full analysis, compute a confusion matrix.

        :param other: Another Annotations object.
        :return: a dictionary containing incorrect label predictions for given spans
        """
        if not isinstance(other, Annotations):
            raise ValueError("An Annotations object is required as an argument.")

        ambiguity_dict = {}

        for label, start, end, text in self.annotations:
            for c_label, c_start, c_end, c_text in other.annotations:
                if label == c_label:
                    continue
                overlap = max(0, min(end, c_end) - max(c_start, start))
                if overlap != 0:
                    ambiguity_dict[(label, start, end, text)] = []
                    ambiguity_dict[(label, start, end, text)].append((c_label, c_start, c_end, c_text))

        return ambiguity_dict

    def compute_confusion_matrix(self, other, entities, leniency=0):
        """
        Computes a confusion matrix representing span level ambiguity between this annotation and the argument annotation.
        An annotation in 'annotations' is ambiguous is it overlaps with a span in this Annotation but does not have the
        same entity label. The main diagonal of this matrix corresponds to entities in this Annotation that match spans
        in 'annotations' and have equivalent class label.

        :param other: Another Annotations object.
        :param entities: a list of entities to use in computing matrix ambiguity.
        :param leniency: leniency to utilize when computing overlapping entities. This is the same definition of leniency as in intersection.
        :return: a square matrix with dimension len(entities) where matrix[i][j] indicates that entities[i] in this annotation was predicted as entities[j] in 'annotation' matrix[i][j] times.
        """
        if not isinstance(other, Annotations):
            raise ValueError("An Annotations object is required as an argument.")
        if not isinstance(entities, list):
            raise ValueError("A list of entities is required")

        entity_encoding = {entity: int(i) for i, entity in enumerate(entities)}
        confusion_matrix = [[0 for x in range(len(entities))] for x in range(len(entities))]

        ambiguity_dict = self.compute_ambiguity(other)
        intersection = self.intersection(other, leniency=leniency)

        # Compute all off diagonal scores
        for gold_span in ambiguity_dict:
            gold_label, start, end, text = gold_span
            for ambiguous_span in ambiguity_dict[gold_span]:
                ambiguous_label, _, _, _ = ambiguous_span
                confusion_matrix[entity_encoding[gold_label]][entity_encoding[ambiguous_label]] += 1

        # Compute diagonal scores (correctly predicted entities with correct spans)
        for matching_span in intersection:
            matching_label, start, end, text = matching_span
            confusion_matrix[entity_encoding[matching_label]][entity_encoding[matching_label]] += 1

        return confusion_matrix

    def compute_counts(self):
        """
        Computes counts of each entity type and relation type in this annotation.
        :return: a dictionary containing counts
        """
        return Counter(e[0] for e in self.annotations)

    def __str__(self):
        return str(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    def __getitem__(self, item):
        return self.annotations[item]
