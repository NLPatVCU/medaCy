# Author : Samantha Mahendran for RelaCy

from medacy.tools import DataFile, Annotations
from operator import itemgetter
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
import spacy


def list_to_file(file, input_list):
    """
    Function to write the contents of a list to a file

    :param file: name of the output file.
    :param input_list: list needs to be written to file
    """
    with open(file, 'w') as f:
        for item in input_list:
            f.write("%s\n" % item)


def remove_Punctuation(string):
    """
    Function to remove punctuation from a given string. It traverses the given string
    and replaces the punctuation marks with null

    :param string: given string.
    :return string:string without punctuation
    """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")

    return string


def replace_Punctuation(string):
    """
    Function to remove punctuation from a given string. It traverses the given string
    and replaces the punctuation marks with comma (,)

    :param string: given string.
    :return string:string without punctuation
    """
    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, ",")

    return string


def add_file_segments(doc_segments, segment):
    """
    Function to add the local segment object to the global segment object
    :param doc_segments: global segment object
    :param segment: local segment object
    :return: doc_segments
    """
    doc_segments['preceding'].extend(segment['preceding'])
    doc_segments['concept1'].extend(segment['concept1'])
    doc_segments['middle'].extend(segment['middle'])
    doc_segments['concept2'].extend(segment['concept2'])
    doc_segments['succeeding'].extend(segment['succeeding'])
    doc_segments['sentence'].extend(segment['sentence'])
    doc_segments['label'].extend(segment['label'])

    return doc_segments


def extract_Segments(sentence, span1, span2):
    """
    Takes a sentence and the span of both entities as the input. First it locates the entities in the sentence and
    divides the sentence into following segments:

    Preceding - (tokenized words before the first concept)
    concept 1 - (tokenized words in the first concept)
    Middle - (tokenized words between 2 concepts)
    concept 2 - (tokenized words in the second concept)
    Succeeding - (tokenized words after the second concept)

    :param sentence: the sentence where both entities exist
    :param span1: span of the first entity
    :param span2: span of the second entity
    :return: preceding, middle, succeeding
    """

    preceding = sentence[0:sentence.find(span1)]
    preceding = remove_Punctuation(str(preceding)).strip()

    middle = sentence[sentence.find(span1) + len(span1):sentence.find(span2)]
    middle = remove_Punctuation(str(middle)).strip()

    succeeding = sentence[sentence.find(span2) + len(span2):]
    succeeding = remove_Punctuation(str(succeeding)).strip()

    return preceding, middle, succeeding


class Segmentation:

    def __init__(self, dataset=None, sentence_align = False):
        self.dataset = dataset
        self.nlp_model = English()
        # self.nlp_model = spacy.load('en_core_web_sm')

        """
        Simple pipeline component, to allow custom sentence boundary detection logic that doesn’t require the dependency parse.
        A simpler, rule-based strategy that doesn’t require a statistical model to be loaded
        """
        if sentence_align:
            sentencizer = Sentencizer(punct_chars=["\n"])
        else:
            sentencizer = Sentencizer(punct_chars=["\n", ".", "?"])

        self.nlp_model.add_pipe(sentencizer)

        # global segmentation object that returns all segments and the label
        self.segments = {'seg_preceding': [], 'seg_concept1': [], 'seg_concept2': [], 'seg_middle': [],
                         'seg_succeeding': [], 'sentence': [], 'label': []}

        for datafile in dataset:
            print(datafile)
            self.ann_path = datafile.get_annotation_path()
            self.txt_path = datafile.get_text_path()
            self.ann_obj = Annotations(self.ann_path)

            content = open(self.txt_path).read()
            # content_text = replace_Punctuation(content)

            self.doc = self.nlp_model(content)

            segment = self.get_Segments_from_sentence(self.ann_obj)
            # segment = self.get_Segments_from_relations(self.ann_obj )

            # Add lists of segments to the segments object for the dataset
            self.segments['seg_preceding'].extend(segment['preceding'])
            self.segments['seg_concept1'].extend(segment['concept1'])
            self.segments['seg_middle'].extend(segment['middle'])
            self.segments['seg_concept2'].extend(segment['concept2'])
            self.segments['seg_succeeding'].extend(segment['succeeding'])
            self.segments['sentence'].extend(segment['sentence'])
            self.segments['label'].extend(segment['label'])

            # To add lists of segments to the segments object for the dataset while maintaining the list separate
            # self.segments['seg_preceding'].append(segment['preceding'])
            # self.segments['seg_preceding'].append(segment['preceding'])
            # self.segments['seg_concept1'].append(segment['concept1'])
            # self.segments['seg_middle'].append(segment['middle'])
            # self.segments['seg_concept2'].append(segment['concept2'])
            # self.segments['seg_succeeding'].append(segment['succeeding'])
            # self.segments['sentence'].append(segment['sentence'])
            # self.segments['label'].append(segment['label'])

        #print the number of instances of each relation classes
        print([(i, self.segments['label'].count(i)) for i in set(self.segments['label'])])

        # write the segments to a file
        list_to_file('sentence_train', self.segments['sentence'])
        list_to_file('preceding_seg', self.segments['seg_preceding'])
        list_to_file('concept1_seg', self.segments['seg_concept1'])
        list_to_file('middle_seg', self.segments['seg_middle'])
        list_to_file('concept2_seg', self.segments['seg_concept2'])
        list_to_file('succeeding_seg', self.segments['seg_succeeding'])
        list_to_file('labels_train', self.segments['label'])

    def get_Segments_from_relations(self, ann):

        """
        For each relation object, it identifies the label and the entities first, then extracts the span of the
        entities from the text file using the start and end character span of the entities. Then it finds the
        sentence the entities are located in and passes the sentence and the spans of the entities to the function
        that extracts the following segments:

        Preceding - (tokenize words before the first concept)
        concept 1 - (tokenize words in the first concept)
        Middle - (tokenize words between 2 concepts)
        concept 2 - (tokenize words in the second concept)
        Succeeding - (tokenize words after the second concept)

        :param ann: annotation object
        :return: segments and label
        """

        # object to store the segments of a relation object
        segment = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                   'label': []}

        for label_rel, entity1, entity2 in ann.annotations['relations']:

            start_C1 = ann.annotations['entities'][entity1][1]
            end_C1 = ann.annotations['entities'][entity1][2]

            start_C2 = ann.annotations['entities'][entity2][1]
            end_C2 = ann.annotations['entities'][entity2][2]

            # to get arrange the entities in the order they are located in the sentence
            if start_C1 < start_C2:
                concept_1 = self.doc.char_span(start_C1, end_C1)
                concept_2 = self.doc.char_span(start_C2, end_C2)
            else:
                concept_1 = self.doc.char_span(start_C2, end_C2)
                concept_2 = self.doc.char_span(start_C1, end_C1)

            # get the sentence where the entity is located
            sentence_C1 = str(concept_1.sent)
            sentence_C2 = str(concept_2.sent)

            # if both entities are located in the same sentence return the sentence or
            # concatenate the individual sentences where the entities are located in to one sentence

            if sentence_C1 == sentence_C2:
                sentence = sentence_C1
            else:
                sentence = sentence_C1 + " " + sentence_C2

            sentence = remove_Punctuation(str(sentence).strip())
            concept_1 = remove_Punctuation(str(concept_1).strip())
            concept_2 = remove_Punctuation(str(concept_2).strip())
            segment['concept1'].append(concept_1)
            segment['concept2'].append(concept_2)
            segment['sentence'].append(sentence.replace('\n', ' '))

            preceding, middle, succeeding = extract_Segments(sentence, concept_1, concept_2)
            segment['preceding'].append(preceding.replace('\n', ' '))
            segment['middle'].append(middle.replace('\n', ' '))
            segment['succeeding'].append(succeeding.replace('\n', ' '))
            segment['label'].append(label_rel)

        return segment

    def get_Segments_from_sentence(self, ann):

        """

        In the annotation object, it identifies the sentence each problem entity is located and tries to determine
        the relations between other problem entities and other entity types in the same sentence. When a pair of
        entities is identified first it checks whether a annotated relation type exists, in that case it labels with
        the given annotated label if not it labels as a No - relation pair. finally it passes the sentence and the
        spans of the entities to the function that extracts the following segments:

        Preceding - (tokenize words before the first concept)
        concept 1 - (tokenize words in the first concept)
        Middle - (tokenize words between 2 concepts)
        concept 2 - (tokenize words in the second concept)
        Succeeding - (tokenize words after the second concept)

        :param ann: annotation object
        :return: segments and label: preceding, concept_1, middle, concept_2, succeeding, label

        """
        # object to store the segments of a relation object for a file
        doc_segments = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                        'label': []}

        # list to store the identified relation pair for problem-problem
        self.entity_holder = []

        for key1, value1 in ann.annotations['entities'].items():
            label1, start1, end1, mention1 = value1

            if label1 == 'problem':
                for key2, value2 in ann.annotations['entities'].items():
                    label2, start2, end2, mention2 = value2
                    token = True

                    # for the test entities
                    if label2 == 'test':
                        for label_rel, entity1, entity2 in ann.annotations['relations']:
                            if key2 == entity1 and key1 == entity2:
                                segment = self.extract_sentences(ann, label_rel, entity1, entity2, True)
                                doc_segments = add_file_segments(doc_segments, segment)
                                token = False
                                break

                        # No problem-test relations
                        label_rel = 'NTeP'
                        segment = self.extract_sentences(ann, label_rel, key1, key2)
                        if segment is not None:
                            doc_segments = add_file_segments(doc_segments, segment)

                    # for the treatment entities
                    if label2 == 'treatment':
                        for label_rel, entity1, entity2 in ann.annotations['relations']:
                            if key2 == entity1 and key1 == entity2:
                                segment = self.extract_sentences(ann, label_rel, entity1, entity2, True)
                                doc_segments = add_file_segments(doc_segments, segment)
                                token = False
                                break

                        # No problem-treatment relations
                        if token:
                            label_rel = 'NTrP'
                            segment = self.extract_sentences(ann, label_rel, key1, key2)
                            if segment is not None:
                                doc_segments = add_file_segments(doc_segments, segment)

                    # for the other problem entities
                    if label2 == 'problem' and key1 != key2:
                        for label_rel, entity1, entity2 in ann.annotations['relations']:
                            if key2 == entity1 and key1 == entity2:
                                segment = self.extract_sentences(ann, label_rel, entity1, entity2, True)
                                doc_segments = add_file_segments(doc_segments, segment)
                                token = False
                                break

                        # No problem-problem relations
                        if token:
                            label_rel = 'NPP'
                            segment = self.extract_sentences(ann, label_rel, key1, key2)
                            if segment is not None:
                                doc_segments = add_file_segments(doc_segments, segment)
        return doc_segments

    def extract_sentences(self, ann, label_rel, entity1, entity2, from_relation=False):
        """
        when the two entities are give as input, it identifies the sentences they are located and determines whether the
        entity pair is in the same sentence or not. if not they combine the sentences if there an annotated relation exist
        and returns None if an annotated relation doesn't exist
        :param ann: annotation object
        :param label_rel: relation type
        :param entity1: first entity in the considered pair
        :param entity2: second entity in the considered pair
        :param from_relation: check for annotated relation in the data
        :return:
        """
        segment = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                   'label': []}
        start_C1 = ann.annotations['entities'][entity1][1]
        end_C1 = ann.annotations['entities'][entity1][2]

        start_C2 = ann.annotations['entities'][entity2][1]
        end_C2 = ann.annotations['entities'][entity2][2]

        # to get arrange the entities in the order they are located in the sentence
        if start_C1 < start_C2:
            concept_1 = self.doc.char_span(start_C1, end_C1)
            concept_2 = self.doc.char_span(start_C2, end_C2)
        else:
            concept_1 = self.doc.char_span(start_C2, end_C2)
            concept_2 = self.doc.char_span(start_C1, end_C1)

        # get the sentence the entity is located
        sentence_C1 = str(concept_1.sent.text)
        sentence_C2 = str(concept_2.sent.text)

        # if both entities are located in the same sentence return the sentence or
        # concatenate the individual sentences where the entities are located in to one sentence
        if from_relation:
            if sentence_C1 == sentence_C2:
                sentence = sentence_C1
            else:
                sentence = sentence_C1 + " " + sentence_C2
        else:
            if sentence_C1 == sentence_C2:
                sentence = sentence_C1
                entity_pair = entity1 + '-' + entity2
                if entity_pair not in self.entity_holder:
                    self.entity_holder.append(entity2 + '-' + entity1)
                else:
                    sentence = None
            else:
                sentence = None

        if sentence is not None:
            sentence = remove_Punctuation(str(sentence).strip())
            concept_1 = remove_Punctuation(str(concept_1).strip())
            concept_2 = remove_Punctuation(str(concept_2).strip())
            preceding, middle, succeeding = extract_Segments(sentence, concept_1, concept_2)

            # remove the next line character in the extracted segment by replacing the '\n' with ' '
            segment['concept1'].append(concept_1.replace('\n', ' '))
            segment['concept2'].append(concept_2.replace('\n', ' '))
            segment['sentence'].append(sentence.replace('\n', ' '))
            segment['preceding'].append(preceding.replace('\n', ' '))
            segment['middle'].append(middle.replace('\n', ' '))
            segment['succeeding'].append(succeeding.replace('\n', ' '))
            segment['label'].append(label_rel)

        return segment
