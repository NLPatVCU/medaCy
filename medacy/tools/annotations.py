import os, logging, tempfile
import spacy
from medacy.tools import BiluoTokenizer
from medacy.tools.converters.con_to_brat import convert_con_to_brat
from medacy.tools.converters.brat_to_con import convert_brat_to_con
from math import ceil

class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass


class Annotations:
    """
    An Annotations object stores all relevant information needed to manage Annotations over a document.

    The Annotation object is utilized by medaCy to structure input to models and output from models.
    This object wraps a dictionary containing two keys at the root level: 'entities' and 'relations'.
    This structured dictionary is designed to interface easily with the BRAT ANN format. The key 'entities' contains
    as a value a dictionary with keys T_1, T2, ... ,TN corresponding each to a single entity. The key 'relations'
    contains a list of tuple relations where the first element of each tuple is the relation type and the last two
    elements correspond to keys in the 'entities' dictionary.
    """
    def __init__(self, annotation_data, annotation_type='ann', source_text_path=None):
        """
        Creates a new Annotation object.

        :param annotation_data: dictionary formatted as described above or alternatively a file_path to an annotation file
        :param annotation_type: a string or None specifying the type or format of annotation_data given
        :param source_text_path: path to the text file from which the annotations were derived; optional for ann files but necessary for conversion to or from con.
        """
        self.supported_file_types = ['ann', 'con']  # change me when new annotation types are supported
        self.source_text_path = source_text_path

        if not (isinstance(annotation_data, (dict, str))):
            raise TypeError("annotation_data must of type dict or str.")

        if isinstance(annotation_data, dict):
            if not ('entities' in annotation_data and isinstance(annotation_data['entities'], dict)):
                raise InvalidAnnotationError("The dictionary annotation_data must contain a key 'entities' "
                                             "corresponding to a list of entity tuples")
            if 'relations' not in annotation_data and isinstance(annotation_data['relations'], list):
                raise InvalidAnnotationError("The dictionary annotation_data must contain a key 'relations' "
                                             "corresponding to a list of entity tuples")
            self.annotations = annotation_data

        if isinstance(annotation_data, str):
            if annotation_type is None:
                raise ValueError("Must specify the type of annotation file representing by annotation_data")
            if not os.path.isfile(annotation_data):
                raise FileNotFoundError("annotation_data is not a valid file path")

            if annotation_type not in self.supported_file_types:
                raise NotImplementedError("medaCy currently only supports %s annotation files"
                                          % (str(self.supported_file_types)))
            if annotation_type == 'ann':
                self.from_ann(annotation_data)
            elif annotation_type == 'con':
                self.from_con(annotation_data)

    def get_labels(self):
        """
        Get the set of labels from this collection of annotations.

        :return: The set of labels.
        """
        labels = set()

        for entity in self.annotations['entities'].values():
            labels.add(entity[0])

        return labels

    def get_entity_annotations(self, return_dictionary=False, format='medacy', nlp=None):
        """
        Returns a list of entity annotation tuples

        :param return_dictionary: returns the dictionary storing the annotation mappings. Useful if also working with relationship extraction
        :return: A list of entities or underlying dictionary of entities
        """
        if return_dictionary:
            return self.annotations['entities']

        if format == 'medacy':
            return [self.annotations['entities'][T] for T in self.annotations['entities'].keys()]
        elif format == 'spacy':
            if not self.source_text_path:
                raise FileNotFoundError("spaCy format requires the source text path")

            with open(self.source_text_path, 'r') as source_text_file:
                source_text = source_text_file.read()

            entities = []

            for annotation in self.annotations['entities'].values():
                entity = annotation[0]
                start = annotation[1]
                end = annotation[2]
                entities.append((start, end, entity))

            return (source_text, {"entities": entities})
        elif format == 'pytorch':
            if not self.source_text_path:
                raise FileNotFoundError("pytorch format requires the source text path")
            elif not nlp:
                raise TypeError('pytorch format requires nlp to be supplied')

            with open(self.source_text_path, 'r') as source_text_file:
                source_text = source_text_file.read()

            entities = []

            for annotation in self.annotations['entities'].values():
                entity = annotation[0]
                start = annotation[1]
                end = annotation[2]
                entities.append((start, end, entity))

            doc = nlp(source_text)
            biluo_tokenizer = BiluoTokenizer(doc)

            tokens = biluo_tokenizer.get_tokens()
            biluo_labels = biluo_tokenizer.get_labels(entities)

            return (tokens, biluo_labels)
        else:
            raise ValueError("'%s' is not a valid annotation format" % format)

    def get_relation_annotations(self):
        """
        Returns a list of entity annotation tuples

        :return: a list of relations
        """
        return self.annotations['relations']


    def add_entity(self, label, start, end, text=""):
        """
        Adds an entity to the Annotations

        :param label: the label of the annotation you are appending
        :param start: the start index in the document of the annotation you are appending.
        :param end: the end index in the document of the annotation you are appending
        :param text: the raw text of the annotation you are appending
        :return:
        """
        self.annotations['entities'].update({'T%i' % (len(self.annotations['entities']) + 1) : (label, start, end, text)})

    def to_ann(self, write_location=None):
        """
        Formats the Annotations object into a string representing a valid ANN file. Optionally writes the formatted
        string to a destination.

        :param write_location: path of location to write ann file to
        :return: returns string formatted as an ann file, if write_location is valid path also writes to that path.
        """
        ann_string = ""
        entities = self.get_entity_annotations(return_dictionary=True)
        for key in sorted(entities.keys(), key=lambda element: int(element[1:])):  # Sorts by entity number
            entity, first_start, last_end, labeled_text = entities[key]
            ann_string += "%s\t%s %i %i\t%s\n" % (key, entity, first_start, last_end, labeled_text.replace('\n', ' '))

        if write_location is not None:
            if os.path.isfile(write_location):
                logging.warning("Overwriting file at: %s", write_location)
            with open(write_location, 'w') as file:
                file.write(ann_string)

        return ann_string

    def from_ann(self, ann_file_path):
        """
        Loads an ANN file given by ann_file

        :param ann_file_path: the system path to the ann_file to load
        :return: annotations object is loaded with the ann file.
        """
        if not os.path.isfile(ann_file_path):
            raise FileNotFoundError("ann_file_path is not a valid file path")
        self.annotations = {'entities': {}, 'relations': []}
        valid_IDs = ['T', 'R', 'E', 'A', 'M', 'N']
        with open(ann_file_path, 'r') as file:
            annotation_text = file.read()
        for line in annotation_text.split("\n"):
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if "\t" not in line:
                raise InvalidAnnotationError("Line chunks in ANN files are separated by tabs, see BRAT guidelines. %s"
                                             % line)
            line = line.split("\t")
            if not line[0][0] in valid_IDs:
                raise InvalidAnnotationError("Ill formated annotation file, each line must contain of the IDs: %s"
                                             % valid_IDs)
            if 'T' == line[0][0]:
                if len(line) == 2:
                    logging.warning("Incorrectly formatted entity line in ANN file (%s): %s", ann_file_path, line)
                tags = line[1].split(" ")
                entity_name = tags[0]
                entity_start = int(tags[1])
                entity_end = int(tags[-1])
                self.annotations['entities'][line[0]] = (entity_name, entity_start, entity_end, line[-1])

            if 'R' == line[0][0]:  # TODO TEST THIS
                tags = line[1].split(" ")
                assert len(tags) == 3, "Incorrectly formatted relation line in ANN file"
                relation_name = tags[0]
                relation_start = tags[1].split(':')[1]
                relation_end = tags[2].split(':')[1]
                self.annotations['relations'].append((relation_name, relation_start, relation_end))
            if 'E' == line[0][0]:
                logging.warning("Event annotations not implemented in medaCy")
            if 'A' == line[0][0] or 'M' == line[0][0]:
                logging.warning("Attribute annotations not implemented in medaCy")
            if 'N' == line[0][0]:
                logging.warning("Normalization annotations are not implemented in medaCy")

    def to_con(self, write_location=None):
        """
        Formats the Annotation object to a valid con file. Optionally writes the string to a specified location.

        :param write_location: Optional path to an output file; if provided but not an existing file, will be created. If this parameter is not provided, nothing will be written to file.
        :return: A string representation of the annotations in the con format.
        """
        if not self.source_text_path:
            raise FileNotFoundError("The annotation can not be converted to the con format without the source text."
                                    " Please provide the path to the source text as source_text_path in the object's"
                                    " constructor.")

        temp_ann_file = tempfile.mktemp()
        with open(temp_ann_file, 'w+') as f:
            self.to_ann(f.name)
            con_text = convert_brat_to_con(f.name, self.source_text_path)

        if write_location:
            if os.path.isfile(write_location):
                logging.warning("Overwriting file at: %s", write_location)
            with open(write_location, 'w+') as f:
                f.write(con_text)

        return con_text

    def from_con(self, con_file_path):
        """
        Converts a con file from a given path to an Annotations object. The conversion takes place through the
        from_ann() method in this class because the indices for the Annotations object must be those used in
        the BRAT format. The path to the source text for the annotations must be defined unless that file exists
        in the same directory as the con file.

        :param con_file_path: path to the con file being converted to an Annotations object.
        """
        ann_from_con = convert_con_to_brat(con_file_path, self.source_text_path)
        temp_ann_file = tempfile.mktemp()
        with open(temp_ann_file, "w+") as f:
            f.write(ann_from_con)
            self.from_ann(f.name)  # must pass the name to self.from_ann() to ensure compatibility

    def difference(self, annotations, leniency=0):
        """
        Identifies the difference between two Annotations objects. Useful for checking if an unverified annotation
        matches an annotation known to be accurate. This is done returning a list of all annotations in the operated on
        Annotation object that do not exist in the passed in annotation object. This is a set difference.

        :param annotations: Another Annotations object.
        :param leniency: a floating point value between [0,1] defining the leniency of the character spans to count as different. A value of zero considers only exact character matches while a positive value considers entities that differ by up to :code:`ceil(leniency * len(span)/2)` on either side.
        :return: A set of tuples of non-matching annotations.
        """
        if not isinstance(annotations, Annotations):
            raise ValueError("Annotations.diff() can only accept another Annotations object as an argument.")
        if leniency != 0:
            if not  0 <= leniency <= 1:
                raise ValueError("Leniency must be a floating point between [0,1]")
        else:
            return set(self.get_entity_annotations()).difference(annotations.get_entity_annotations())

        matches = set()
        for label, start, end, text in self.get_entity_annotations():
            window = ceil(leniency * (end-start))
            for c_label, c_start, c_end, c_text in annotations.get_entity_annotations():
                if label == c_label:
                    if start - window <= c_start and end+window >= c_end:
                        matches.add((label, start, end, text))
                        break


        return set(self.get_entity_annotations()).difference(matches)

    def intersection(self, annotations, leniency=0):
        """
        Computes the intersection of the operated annotation object with the operand annotation object.

        :param annotations: Another Annotations object.
        :param leniency: a floating point value between [0,1] defining the leniency of the character spans to count as different. A value of zero considers only exact character matches while a positive value considers entities that differ by up to :code:`ceil(leniency * len(span)/2)` on either side.
        :return A set of annotations that appear in both Annotation objects
        """
        if not isinstance(annotations, Annotations):
            raise ValueError("An Annotations object is requried as an argument.")
        if leniency != 0:
            if not  0 <= leniency <= 1:
                raise ValueError("Leniency must be a floating point between [0,1]")

        matches = set()
        for label, start, end, text in self.get_entity_annotations():
            window = ceil(leniency * (end-start))
            for c_label, c_start, c_end, c_text in annotations.get_entity_annotations():
                if label == c_label:
                    if start - window <= c_start and end+window >= c_end:
                        matches.add((label, start, end, text))
                        break


        return matches

    def compute_ambiguity(self, annotations):
        """
        Finds occurrences of spans from 'annotations' that intersect with a span from this annotation but do not have this spans label.
        label. If 'annotation' comprises a models predictions, this method provides a strong indicators
        of a model's in-ability to dis-ambiguate between entities. For a full analysis, compute a confusion matrix.

        :param annotations: Another Annotations object.
        :return: a dictionary containing incorrect label predictions for given spans
        """
        if not isinstance(annotations, Annotations):
            raise ValueError("An Annotations object is required as an argument.")

        ambiguity_dict = {}

        for label, start, end, text in self.get_entity_annotations():
            for c_label, c_start, c_end, c_text in annotations.get_entity_annotations():
                if label == c_label:
                    continue
                overlap = max(0, min(end, c_end) - max(c_start, start))
                if overlap != 0:
                    ambiguity_dict[(label, start, end, text)] = []
                    ambiguity_dict[(label, start, end, text)].append((c_label, c_start, c_end, c_text))

        return ambiguity_dict

    def compute_confusion_matrix(self, annotations, entities, leniency=0):
        """
        Computes a confusion matrix representing span level ambiguity between this annotation and the argument annotation.
        An annotation in 'annotations' is ambigous is it overlaps with a span in this Annotation but does not have the
        same entity label. The main diagonal of this matrix corresponds to entities in this Annotation that match spans
        in 'annotations' and have equivalent class label.

        :param annotations: Another Annotations object.
        :param entities: a list of entities to use in computing matrix ambiguity.
        :param leniency: leniency to utilize when computing overlapping entities. This is the same definition of leniency as in intersection.
        :return: a square matrix with dimension len(entities) where matrix[i][j] indicates that entities[i] in this annotation was predicted as entities[j] in 'annotation' matrix[i][j] times.
        """
        if not isinstance(annotations, Annotations):
            raise ValueError("An Annotations object is required as an argument.")
        if not isinstance(entities, list):
            raise ValueError("A list of entities is required")

        entity_encoding = {entity: int(i) for i, entity in enumerate(entities)}
        confusion_matrix = [[0 for x in range(len(entities))] for x in range(len(entities))]

        ambiguity_dict = self.compute_ambiguity(annotations)
        intersection = self.intersection(annotations, leniency=leniency)

        #Compute all off diagonal scores
        for gold_span in ambiguity_dict:
            gold_label, start, end, text = gold_span
            for ambiguous_span in ambiguity_dict[gold_span]:
                ambiguous_label, _, _, _ = ambiguous_span
                confusion_matrix[entity_encoding[gold_label]][entity_encoding[ambiguous_label]] += 1

        #Compute diagonal scores (correctly predicted entities with correct spans)
        for matching_span in intersection:
            matching_label, start, end, text = matching_span
            confusion_matrix[entity_encoding[matching_label]][entity_encoding[matching_label]] += 1

        return confusion_matrix

    def compute_counts(self):
        """
        Computes counts of each entity type and relation type in this annotation.

        :return: a dictionary containing counts
        """

        counts = {
            'entities':{},
            'relations':{}
        }

        for entity,_,_,_ in self.get_entity_annotations():
            counts['entities'][entity] = counts['entities'].get(entity, 0) + 1
        for relation,_,_ in self.get_relation_annotations():
            counts['relations'][relation] = counts['relations'].get(relation, 0) + 1

        return counts

    def __str__(self):
        return str(self.annotations)
