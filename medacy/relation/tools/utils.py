#Author : Samantha Mahendran for RelaCy

from medacy.tools import DataFile, Annotations
import spacy

class Utils:

    def __init__(self, dataset = None):
        self.dataset = dataset
        self.nlp_model = spacy.load('en_core_web_sm')
        for datafile in dataset:
            self.datafile = datafile
            self.ann_path = datafile.get_annotation_path()
            self.txt_path = datafile.get_text_path()
            self.ann_obj = Annotations(self.ann_path)

            content = open(self.txt_path).read()
            self.doc = self.nlp_model(content)
            self.find_mismatchSpan()


    def count_RelationClasses (self):

        rel_TrIP = 0
        rel_TrWP = 0
        rel_TrCP = 0
        rel_TrAP = 0
        rel_TrNAP = 0
        rel_NTrP = 0
        rel_TeRP = 0
        rel_TeCP = 0
        rel_NTeP = 0
        rel_PIP = 0
        rel_NPP = 0

        for label_rel, entity1, entity2 in self.ann_obj.annotations['relations']:
            if label_rel == "TrIP":
                rel_TrIP += 1
            elif label_rel == "TrWP":
                rel_TrWP += 1
            elif label_rel == "TrCP":
                rel_TrCP += 1
            elif label_rel == "TrAP":
                rel_TrAP += 1
            elif label_rel == "TrNAP":
                rel_TrNAP += 1
            elif label_rel == "NTrP":
                rel_NTrP += 1
            elif label_rel == "TeRP":
                rel_TeRP += 1
            elif label_rel == "TeCP":
                rel_TeCP += 1
            elif label_rel == "NTeP":
                rel_NTeP += 1
            elif label_rel == "PIP":
                rel_PIP += 1
            else:
                rel_NPP +=1
        total_labels = rel_TrIP + rel_TrWP + rel_TrCP + rel_TrAP + rel_TrNAP + rel_NTrP + rel_TeRP + rel_TeCP + rel_NTeP + rel_PIP  + rel_NPP

        print("Relation class statistics:")
        print("TrIP: ", rel_TrIP)
        print("TrWP: ", rel_TrWP)
        print("TrCP: ", rel_TrCP)
        print("TrAP: ", rel_TrAP)
        print("TrNAP: ", rel_TrNAP)
        print("NTrP: ", rel_NTrP)
        print("TeRP: ", rel_TeRP)
        print("TeCP: ", rel_TeCP)
        print("NTeP: ", rel_NTeP)
        print("PIP: ", rel_PIP)
        print("NPP: ", rel_NPP)

        print ("Total number of relations :", total_labels)

    def find_mismatchSpan (self):
        for label, start, end, span in [self.ann_obj.annotations['entities'][key] for key in self.ann_obj.annotations['entities']]:
            span_doc = self.doc.char_span(start, end)
            span_doc = str(span_doc).lower().replace('\n', ' ')
            if str(span).lower()!= span_doc:
                print("File : ", self.datafile)
                print("Mismatch:", label, start, end, str(span).lower(), span_doc)