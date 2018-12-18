import os
"""
A medaCy annotation. This stores all relevent information needed as input to medaCy or as output
"""

class Annotations:

    def __init__(self, annotation_data, annotation_type='ann'):
        """
        Created an Annotation object utilized by medaCy to structure input to models and output from models.
        This object wraps a dictionary containing two keys at the root level: 'entities' and 'relations'.
        This structured dictionary is designed to interface easily with the BRAT ANN format. The key 'entities' contains
        as a value a dictionary with keys T1, T2, ... ,TN corresponding each to a single entity. The key 'relations'
        contains a list of tuple relations where the first element of each tuple is the relation type and the last two
        elements correspond to keys in the 'entities' dictionary.

        :param annotation_data: dictionary formatted as described above or alternatively a file_path to an annotation file
        :param annotation_type: a string or None specifying the type or format of annotation_data given
        """
        self.supported_file_types = ['ann'] #change me when new annotation types are supported

        assert isinstance(annotation_data, dict) or isinstance(annotation_data, str),\
            "annotation_data must of type dict or str"

        if isinstance(annotation_data, dict):
            assert 'entities' in annotation_data and isinstance(annotation_data['entities'], dict), \
                "The dictionary annotation_data must contain a key 'entities' corresponding to a dictionary of entities"
            assert 'relations' in annotation_data and isinstance(annotation_data['relations'], list), \
                "The dictionary annotation_data must contain a key 'relations corresponding to a list of entity tuples"
            self.annotations = annotation_data


        if isinstance(annotation_data, str):
            assert annotation_type is not None, "Must specify the type of annotation file representing by annotation_data"
            assert os.path.isfile(annotation_data), "annotation_data is not a valid file path"

            assert annotation_type in self.supported_file_types, \
                "medaCy currently only supports %s annotation files" % (str(self.supported_file_types))
            if annotation_type == 'ann':
                self._load_ann(annotation_data)

    def contains_entities(self):
        return self.annotations['entities']

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

    def _load_ann(self, ann_file):
        """
        Loads an ANN file and adhering to BRAT guidelines (http://brat.nlplab.org/standoff.html)
        :param ann_file: the system path to the ann_file to load
        :return: annotations object is loaded with the ann file.
        """
        assert os.path.isfile(ann_file), "ann_file is not a valid file path"
        self.annotations = {'entities': {}, 'relations': []}
        valid_IDs = ['T', 'R', 'E', 'A','M' 'N']
        with open(ann_file, 'r') as file:
            annotation_text = file.read()
            for line in annotation_text.split("\n"):
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                assert "\t" in line, "Line chunks in ANN files are seperated by tabs, see BRAT guidelines. %s" % line
                line = line.split("\t")
                assert line[0][0] == 'T' or line[0][0] =='R',\
                    "Ill formated annotation file, each line must contain of the IDs: %s" % valid_IDs
                if 'T' == line[0][0]:
                    tags = line[1].split(" ")
                    assert len(tags) == 3, "Incorrectly formatted entity line in ANN file"
                    entity_name = tags[0]
                    entity_start = int(tags[1])
                    entity_end = int(tags[-1])
                    self.annotations['entities'][line[0]] = (entity_start, entity_end, entity_name)

                if 'R' == line[0][0]:
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




    def __str__(self):
        return str(self.annotations)


