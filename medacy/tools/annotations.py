import os, logging, tempfile
from medacy.tools.con.con_to_brat import convert_con_to_brat
from medacy.tools.con.brat_to_con import convert_brat_to_con


class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass


class Annotations:
    """
    A medaCy annotation. This stores all relevant information needed as input to medaCy or as output.
    The Annotation object is utilized by medaCy to structure input to models and output from models.
    This object wraps a dictionary containing two keys at the root level: 'entities' and 'relations'.
    This structured dictionary is designed to interface easily with the BRAT ANN format. The key 'entities' contains
    as a value a dictionary with keys T1, T2, ... ,TN corresponding each to a single entity. The key 'relations'
    contains a list of tuple relations where the first element of each tuple is the relation type and the last two
    elements correspond to keys in the 'entities' dictionary.
    """

    def __init__(self, annotation_data, annotation_type='ann', source_text_path=None):
        """
        :param annotation_data: dictionary formatted as described above or alternatively a file_path to an annotation file
        :param annotation_type: a string or None specifying the type or format of annotation_data given
        :param source_text_path: path to the text file from which the annotations were derived; optional for ann files
            but necessary for conversion to or from con.
        """
        self.supported_file_types = ['ann', 'con']  # change me when new annotation types are supported
        self.source_text_path = source_text_path

        if not (isinstance(annotation_data, dict) or isinstance(annotation_data, str)):
            raise InvalidAnnotationError("annotation_data must of type dict or str.")

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
                raise InvalidAnnotationError("Must specify the type of annotation file representing by annotation_data")
            if not os.path.isfile(annotation_data):
                raise FileNotFoundError("annotation_data is not a valid file path")

            if annotation_type not in self.supported_file_types:
                raise NotImplementedError("medaCy currently only supports %s annotation files"
                                             % (str(self.supported_file_types)))
            if annotation_type == 'ann':
                self.from_ann(annotation_data)
            elif annotation_type == 'con':
                self.from_con(annotation_data)

    def get_entity_annotations(self, return_dictionary=False):
        """
        Returns a list of entity annotation tuples
        :param return_dictionary: returns the dictionary storing the annotation mappings. Useful if also working with
        relationship extraction
        :return: a list of entities or underlying dictionary of entities
        """
        if return_dictionary:
            return self.annotations['entities']
        return [self.annotations['entities'][T] for T in self.annotations['entities'].keys()]

    def get_entity_count(self):
        return len(self.annotations['entities'].keys())

    def to_ann(self, write_location=None):
        """
        Formats the Annotations object into a string representing a valid ANN file. Optionally writes the formatted
        string to a destination.
        :param write_location: path of location to write ann file to
        :return: returns string formatted as an ann file, if write_location is valid path also writes to that path.
        """
        ann_string = ""
        entities = self.get_entity_annotations(return_dictionary=True)
        for key in sorted(entities.keys(), key= lambda element: int(element[1:])): #Sorts by entity number
            entity, first_start, last_end, labeled_text = entities[key]
            ann_string += "%s\t%s %i %i\t%s\n" % (key, entity, first_start, last_end, labeled_text.replace('\n', ' '))

        if write_location is not None:
            if os.path.isfile(write_location):
                logging.warning("Overwriting file at: %s", write_location)
            with open(write_location, 'w') as file:
                file.write(ann_string)

        return ann_string

    def from_ann(self, ann_file):
        """
        Loads an ANN file given by ann_file
        :param ann_file: the system path to the ann_file to load
        :return: annotations object is loaded with the ann file.
        """
        if not os.path.isfile(ann_file):
            raise FileNotFoundError("ann_file is not a valid file path")
        self.annotations = {'entities': {}, 'relations': []}
        valid_IDs = ['T', 'R', 'E', 'A', 'M', 'N']
        with open(ann_file, 'r') as file:
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
                    logging.warning("Incorrectly formatted entity line in ANN file (%s): %s", ann_file, line)
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
                raise NotImplementedError("Event annotations not implemented in medaCy")
            if 'A' == line[0][0] or 'M' == line[0][0]:
                raise NotImplementedError("Attribute annotations are not implemented in medaCy")
            if 'N' == line[0][0]:
                raise NotImplementedError("Normalization annotations are not implemented in medaCy")

    def to_con(self, write_location=None):
        """
        Formats the Annotation object to a valid con file. Optionally writes the string to a specified location.
        :param write_location: Optional path to an output file; if provided but not an existing file, will be
            created. If this parameter is not provided, nothing will be written to file.
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
            with open(write_location, 'w+') as file:
                file.write(con_text)

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

    def diff(self, other_anno):
        """
        Identifies the difference between two Annotations objects. Useful for checking if an unverified annotation
        matches an annotation known to be accurate.
        :param other_anno: Another Annotations object.
        :return: A list of tuples of non-matching annotation pairs.
        """
        if not isinstance(other_anno, Annotations):
            raise ValueError("Annotations.diff() can only accept another Annotations object as an argument.")

        these_entities = list(self.annotations['entities'].values())
        other_entities = list(other_anno.annotations['entities'].values())

        if these_entities.__len__() != other_entities.__len__():
            raise ValueError("These annotations cannot be compared because they contain a different number of entities.")

        non_matching_annos = []

        for i in range(0, these_entities.__len__()):
            if these_entities[i] != other_entities[i]:
                non_matching_annos.append(tuple(these_entities[i], other_entities[i]))

        return non_matching_annos

    def __str__(self):
        return str(self.annotations)
  
