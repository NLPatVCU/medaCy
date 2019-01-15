"""
:author: Andriy Mulyar, Steele W. Farnsworth
:date: 12 January, 2019
"""

import os, logging, tempfile
from medacy.tools.con.con_to_brat import convert_con_to_brat
from medacy.tools.con.brat_to_con import convert_brat_to_con
from math import floor, ceil
import numpy as np
from spacy.displacy import EntityRenderer
from statistics import mean


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

    __default_strict = 0.2  # Used in compare_by_index() and indirectly in compare_by_index_stats()

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
                non_matching_annos.append((these_entities[i], other_entities[i]))

        return non_matching_annos

    def compare_by_entity(self, gold_anno):
        """
        Compares two Annotations for checking if an unverified annotation matches an accurate one by creating a data
        structure that looks like this:

        {
            'females': {
                'this_anno': [('Sex', 1396, 1403), ('Sex', 295, 302), ('Sex', 3205, 3212)],
                'gold_anno': [('Sex', 1396, 1403), ('Sex', 4358, 4365), ('Sex', 263, 270)]
                }
            'SALDOX': {
                'this_anno': [('GroupName', 5408, 5414)],
                'gold_anno': [('TestArticle', 5406, 5412)]
                }
            'MISSED_BY_PREDICTION':
                [('GroupName', 8644, 8660, 'per animal group'), ('CellLine', 1951, 1968, 'on control diet (')]
        }

        The object itself should be the predicted Annotations and the argument should be the gold Annotations.

        :param gold_anno: the Annotations object for the gold data.
        :return: The data structure detailed above.
        """
        if not isinstance(gold_anno, Annotations):
            raise ValueError("Annotations.compare_by_entity() can only accept another Annotations object as an argument.")

        these_entities = list(self.annotations['entities'].values())
        gold_entities = list(gold_anno.annotations['entities'].values())

        comparison = {"MISSED_BY_PREDICTION": []}

        for e in these_entities:
            entity = e[3]
            # In this context, an annotation (lowercase) is the entity type and indices for an entity
            entity_annotation = tuple(e[0:3])

            # Create a key for each unique entity, regardless of how many times it appears in the Annotations
            if entity not in comparison.keys():
                comparison[entity] = {"this_anno": [entity_annotation], "gold_anno": []}
            # If there's already a key for a matching entity, add it to the list of annotations for that entity
            else:
                comparison[entity]["gold_anno"].append(entity_annotation)

        for e in gold_entities:
            entity = e[3]
            entity_annotation = tuple(e[0:3])

            if entity in comparison.keys():
                comparison[entity]["gold_anno"].append(entity_annotation)
            else:
                comparison["MISSED_BY_PREDICTION"].append(e)

        return comparison

    def compare_by_index(self, gold_anno, strict=__default_strict):
        """
        Similar to compare_by_entity, but organized by start index. The two data sets used in the comparison will often
        not have two annotations beginning at the same index, so the strict value is used to calculate within what
        margin a matched pair can be separated.
        :param gold_anno: The Annotation object representing an annotation set that is known to be accurate.
        :param strict: Used to calculate within what range a possible match can be. The length of the entity is
            multiplied by this number, and the product of those two numbers is the difference that the entity can
            begin or end relative to the starting index of the entity in the gold dataset. Default is 0.2.
        :return:
        """

        # Guarder conditions
        if not isinstance(gold_anno, Annotations):
            raise ValueError("Annotations.compare_by_index() can only accept another Annotations object "
                             "as an argument.")
        if not isinstance(strict, (int, float)):
            raise ValueError("strict must be an int or float.")
        if strict < 0: raise ValueError("strict must be above 0.")

        def find_closest_key(target: int, matches: list):
            """
            Used to approximate which entity in the predicted data matches a given entity in the gold data when
            they do not have the same start index. (For example, there might be a leading punctuation mark
            in one of the entities.)

            Finds which match (from a list of matches, all ints) is closest to the target (also an int).
            Used to map entities in the predicted data set to the closest entity in the gold dataset.
            :param target: The key of an entity in the predicted dataset.
            :param matches: A list of keys in the gold dataset.
            :return: The key in the list of matches closest to the target.
            """
            matches_array = np.array(matches)
            closest_ind = (np.abs(matches_array - target)).argmin()
            return matches[closest_ind]

        def calculate_accuracy(gold_start, gold_end, pred_start, pred_end) -> float:
            """
            Calculates how closely the start and end indices of the predicted data match the start and end indices of
            the gold data.
            :param gold_start: The start index of the gold data.
            :param gold_end: The end index of the gold data.
            :param pred_start: The start index of the preidcted data.
            :param pred_end: The end index of the predicted data.
            :return: A float representing the percentage accuracy between 0 and 1.
            """
            start_difference_span = abs(gold_start - pred_start)
            start_difference = abs(gold_start - start_difference_span)
            start_accuracy = start_difference / gold_start

            end_difference_span = abs(gold_end - pred_end)
            end_difference = abs(gold_end - end_difference_span)
            end_accuracy = end_difference / gold_end

            overall_accuracy = (start_accuracy + end_accuracy) / 2
            return overall_accuracy

        these_entities = list(self.annotations['entities'].values())
        gold_entities = list(gold_anno.annotations['entities'].values())

        comparison = {"NOT_MATCHED": []}

        for e in gold_entities:
            # start_ind must be an int for later calculations using the indices
            start_ind = int(e[1])
            comparison[start_ind] = {"gold_anno": e, "this_anno": None, "accuracy": 0}

        for e in these_entities:
            start_ind = int(e[1])
            entity = e[3]

            # If there's an exact match for a predicted start index and a gold start index:
            if start_ind in comparison.keys():
                comparison[start_ind]["this_anno"] = e
                comparison[start_ind]["accuracy"] = 1
            else:  # To find the closest key when there is not a 1:1 correlation:
                # Create a range of indices that the match could be in, determined by the strict value
                margin = len(entity) * strict
                range_min = floor(start_ind - margin)
                range_max = ceil(start_ind + margin)
                # Get all the keys that are in the range; "NOT_MATCHED" is a key but is implicitly excluded because
                # strings are not in a range of ints.
                possible_keys = [k for k in comparison.keys() if k in range(range_min, range_max)]
                # find_closest_key() is not very quick, so it's only called if there's more than one possible key.
                if len(possible_keys) > 1:
                    best_key = find_closest_key(start_ind, possible_keys)
                    matched_relation = comparison[best_key]
                    # Take note if the unlikely event occurs that two entities in the predicted data are matched to the
                    # same entity in the gold dataset
                    if matched_relation["this_anno"] is not None:
                        logging.log("Writing over previously matched entity at index %i" % best_key)
                    matched_relation["this_anno"] = e
                    gold_data = matched_relation["gold_anno"]
                    matched_relation["accuracy"] = calculate_accuracy(gold_data[1], gold_data[2], e[1], e[2])
                elif len(possible_keys) == 1:
                    matching_relation = comparison[possible_keys[0]]
                    matching_relation["this_anno"] = e
                    gold_data = matching_relation["gold_anno"]
                    matching_relation["accuracy"] = calculate_accuracy(gold_data[1], gold_data[2], e[1], e[2])
                else:
                    comparison["NOT_MATCHED"].append(e)

        return comparison

    def compare_by_index_stats(self, gold_anno, strict=__default_strict):
        """
        Runs compare_by_index() and returns a dict of related statistics.
        :param gold_anno: See compare_by_index()
        :param strict: See compare_by_index()
        :return: A dictionary with keys:
            "num_not_matched": The number of entites in the predicted data that are not matched to an entity in the
                gold data,
            "avg_accuracy": The average of all the decimal values representing how close to a 1:1 correlation there was
                between the start and end indices in the gold and predicted data.
        """

        if not isinstance(gold_anno, Annotations):
            raise ValueError("Annotations.compare_by_index_stats() can only accept another Annotations object "
                             "as an argument.")
        # Other conditions will be checked when compare_by_index() is called

        # Initialze the object that will be returned
        stats = {"num_not_matched": 0, "avg_accuracy": 0}

        comparison = self.compare_by_index(gold_anno, strict)
        not_matched = comparison.pop("NOT_MATCHED")
        stats["num_not_matched"] = len(not_matched)

        all_avgs = [a["accuracy"] for a in comparison.values()]
        stats["avg_accuracy"] = mean(all_avgs)

        return stats

    def stats(self):
        """
        Count the number of instances of a given entity type and the number of unique entities.
        :return: a dict with keys:
            "entity_counts": a dict matching entities to the number of times that entity appears
                in the Annotations,
            "unique_entity_num": an int of how many unique entities are in the Annotations,
            "entity_list": a list of all the entities that appear in the list; each only appears once.
        """

        # Create a list of entities from a list of annotation tuples
        entities = [i[0] for i in self.annotations['entities'].values()]

        stats = {"entity_counts": {}, "unique_entity_num": 0, "entitiy_list": []}

        entity_counts = stats["entity_counts"]
        unique_entity_num = stats["unique_entity_num"]

        for e in entities:
            if e not in entity_counts.keys():
                entity_counts[e] = 1
                unique_entity_num += 1
            else:
                entity_counts[e] += 1

        stats["entity_list"] = stats["entity_counts"].keys()

        return stats

    def to_html(self, output_file_path, title="medaCy"):
        """
        Convert the Annotations to a displaCy-formatted HTML representation. The Annotations must have the path
        to the source file as one of its attributes. Does not return a value.
        :param output_file_path: Where to write the HTML to.
        :param title: What should appear in the header of the outputted HTML file; not very important
        """

        if self.source_text_path is None:
            raise ValueError("to_html() can only be run on objects for which source_text_path is defined; this instance"
                             " of Annotations was not created with its source_text_path defined.")

        # Instantiate the EntityRenderer with a custom color scheme
        # Only contains entities found in the golden TAC dataset with some colors used twice
        color_scheme = {'SEX': '#7aecec', 'STRAIN': '#bfeeb7', 'SPECIES': '#feca74',
                        'TESTARTICLE': '#ff9561', 'ENDPOINT': '#aa9cfc', 'ENDPOINTUNITOFMEASURE': '#c887fb',
                        'GROUPNAME': '#9cc9cc', 'DOSEROUTE': '#ffeb80', 'DOSE': '#ff8197',
                        'DOSEUNITS': '#ff8197', 'VEHICLE': '#f0d0ff',
                        'TIMEATFIRSTDOSE': '#bfe1d9', 'TIMEATDOSE': '#bfe1d9', 'TIMEATLASTDOSE': '#e4e7d2',
                        'TIMEUNITS': '#e4e7d2', 'TIMEENDPOINTASSESSED': '#e4e7d2',
                        'GROUPSIZE': '#e4e7d2', 'TESTARTICLEPURITY': '#e4e7d2',
                        'SAMPLESIZE': '#7aecec', 'DOSEDURATION': '#bfeeb7', 'DOSEDURATIONUNITS': '#feca74',
                        'DOSEFREQUENCY': '#ff9561', 'CELLLINE': '#aa9cfc', 'TESTARTICLEVERIFICATION': '#c887fb'}
        er = EntityRenderer(options={"colors": color_scheme})

        # EntityRenderer must be passed a list of dictionaries in the format below. This section
        # reformats the internal entity tuples into that format.
        entity_tuples = self.get_entity_annotations()
        displacy_list = []
        for e in entity_tuples:
            displacy_dict = {"start": int(e[1]), "end": int(e[2]), "label": e[0]}
            displacy_list.append(displacy_dict)

        # Get a string of the source text
        with open(self.source_text_path, 'r') as f:
            source_text = f.read()
        # Do the actual HTML rendering
        html = er.render_ents(source_text, displacy_list, title)
        # Write it to file
        with open(output_file_path, 'w+') as f:
            f.write(html)

    def __str__(self):
        return str(self.annotations)
