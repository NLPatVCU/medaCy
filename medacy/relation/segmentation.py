#Author : Samantha Mahendran for RelaCy

from medacy.tools import DataFile, Annotations
import spacy

class Segmentation:

    def __init__(self, dataset = None):
        self.dataset = dataset
        self.nlp_model = spacy.load('en_core_web_sm')

        # segmentation object that returns all segments and the label
        self.segments = {'seg_preceeding': [], 'seg_concept1': [], 'seg_concept2': [], 'seg_middle': [], 'seg_succeeding': [], 'sentence': [], 'label': []}

        for datafile in dataset:
            self.ann_path = datafile.get_annotation_path()
            self.txt_path = datafile.get_text_path()
            self.ann_obj = Annotations(self.ann_path)

            #
            content = open(self.txt_path).read()
            self.doc = self.nlp_model(content)
            segment = self.get_Segments(self.ann_obj )

            #Add lists of segments to the segments object for the dataset
            self.segments['seg_preceeding'].extend(segment['preceeding'])
            self.segments['seg_concept1'].extend(segment['concept1'])
            self.segments['seg_middle'].extend(segment['middle'])
            self.segments['seg_concept2'].extend(segment['concept2'])
            self.segments['seg_succeeding'].extend(segment['succeeding'])
            self.segments['sentence'].extend(segment['sentence'])
            self.segments['label'].extend(segment['label'])

            #To add lists of segments to the segments object for the dataset while maintaining the list separate
            # self.segments['seg_preceeding'].append(segment['preceeding'])
            # self.segments['seg_preceeding'].append(segment['preceeding'])
            # self.segments['seg_concept1'].append(segment['concept1'])
            # self.segments['seg_middle'].append(segment['middle'])
            # self.segments['seg_concept2'].append(segment['concept2'])
            # self.segments['seg_succeeding'].append(segment['succeeding'])
            # self.segments['sentence'].append(segment['sentence'])
            # self.segments['label'].append(segment['label'])

        #write the segments to a file
        self.list_to_file('sentence_train', self.segments['sentence'])
        # self.list_to_file('preceeding_seg', self.segments['seg_preceeding'])
        # self.list_to_file('concept1_seg', self.segments['seg_concept1'])
        # self.list_to_file('middle_seg', self.segments['seg_middle'])
        # self.list_to_file('concept2_seg', self.segments['seg_concept2'])
        # self.list_to_file('suceeding_seg', self.segments['seg_succeeding'])
        self.list_to_file('labels_train', self.segments['label'])


    def remove_Punctuation(self, string):
        """
        Function to remove punctuation from a given string. It traverse the given string
        and if any punctuation marks occur replace it with null

        :param string: given string.
        :return string:string without punctuation
        """
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        for x in string.lower():
            if x in punctuations:
                string = string.replace(x, "")

        return string

    def list_to_file(self, file, list):
        """
        Function to write the contents of a list to a file

        :param file: name of the output file.
        :param list: list needs to be written to file
        """
        with open(file, 'w') as f:
            for item in list:
                f.write("%s\n" % item)

    def get_Segments(self, ann):

        """
        For each relation object, it identifies the label and the entities first, then extracts the span of the entities
        from the text file using the start and end character span of the entities. Then it finds the sentence the entities are located in and passes the
        sentence and the spans of the entities to the function that extracts the following segments:

        Preceding - (tokenized words before the first concept)
        concept 1 - (tokenized words in the first concept)
        Middle - (tokenized words between 2 concepts)
        concept 2 - (tokenized words in the second concept)
        Succeeding - (tokenized words after the second concept)

        :return segments and label: preceeding, concept_1, middle, concept_2, succeeding, label
        """

        #object to store the segments of a relation object
        segment = {'preceeding': [], 'concept1': [], 'concept2': [], 'middle': [],'succeeding': [], 'sentence': [], 'label': []}

        for label_rel, entity1, entity2 in ann.annotations['relations']:
            # if label_rel

            start_C1 = ann.annotations['entities'][entity1][1]
            end_C1 = ann.annotations['entities'][entity1][2]

            start_C2 = ann.annotations['entities'][entity2][1]
            end_C2 = ann.annotations['entities'][entity2][2]

            #to get arrange the entitites in the order they are located in the sentence
            if start_C1 < start_C2:
                concept_1 = self.doc.char_span(start_C1, end_C1)
                concept_2 = self.doc.char_span(start_C2, end_C2)
            else:
                concept_1 = self.doc.char_span(start_C2, end_C2)
                concept_2 = self.doc.char_span(start_C1, end_C1)

            #get the sentence the entity is located
            sentence_C1 = str(concept_1.sent)
            sentence_C2 = str(concept_2.sent)

            #if both enetities are located in the same sentence return the sentence or
            # concatenate the individual sentences where the entities are located in to one sentence

            if (sentence_C1 == sentence_C2):
                sentence = sentence_C1
            else:
                sentence = sentence_C1 + " " + sentence_C2

            sentence = self.remove_Punctuation(str(sentence).strip())
            concept_1 = self.remove_Punctuation(str(concept_1).strip())
            concept_2 = self.remove_Punctuation(str(concept_2).strip())
            segment['concept1'].append(concept_1)
            segment['concept2'].append(concept_2)
            segment['sentence'].append(sentence.replace('\n', ' '))

            preceeding, middle, succeeding = self.extract_Segments(sentence, concept_1, concept_2)
            segment['preceeding'].append(preceeding)
            segment['middle'].append(middle)
            segment['succeeding'].append(succeeding)
            segment['label'].append(label_rel)
        return segment

    def extract_Segments(self, sentence, span1, span2):

        """
        Takes a sentence and the span of both entities as the input. First it locates the entities in the sentence and divides the sentence
        into following segments:

        Preceding - (tokenized words before the first concept)
        concept 1 - (tokenized words in the first concept)
        Middle - (tokenized words between 2 concepts)
        concept 2 - (tokenized words in the second concept)
        Succeeding - (tokenized words after the second concept)

        :return segments: preceeding, middle, succeeding
        """

        preceeding = sentence[0:sentence.find(span1)]
        preceeding = self.remove_Punctuation(preceeding)

        middle = sentence[sentence.find(span1) + len(span1):sentence.find(span2)]
        middle = self.remove_Punctuation(middle)

        succeeding = sentence[sentence.find(span2) + len(span2):]
        succeeding = self.remove_Punctuation(succeeding)

        return preceeding, middle, succeeding



