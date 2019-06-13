import spacy
from spacy.gold import biluo_tags_from_offsets

class BiluoTokenizer:
    doc = None
    
    def __init__(self, text):
        nlp = spacy.load('en_core_web_sm')
        self.doc = nlp(text)

    def get_tokens(self):
        return [str(token) for token in self.doc]

    def fix_entities(self, entities, index):
        fixed_entity = False

        doc = self.doc
        token = doc[index]
        start = token.idx
        end = token.idx + len(token.text)

        print('Looking for an entity that splits (%d, %d, %s)' % (start, end, token.text))

        for i, entity in enumerate(entities):
            # If entity start cuts off token, shift start to beginning of token
            if entity[0] > start and entity[0] <= end:
                print('Original entity ' + str(entity))
                entity = (start, entity[1], entity[2])
                print('Fixed entity' + str(entity))
                fixed_entity = True

            # If entity end cuts off token, shift end to end of token
            if entity[1] >= start and entity[1] < end:
                print('Original entity ' + str(entity))
                entity = (entity[0], end, entity[2])
                print('Fixed entity' + str(entity))
                fixed_entity = True

            entities[i] = entity

        if not fixed_entity:
            return self.fix_entities(entities, index + 1)
        else:
            return entities

    def get_labels(self, entities):
        doc = self.doc
        biluo_labels = biluo_tags_from_offsets(doc, entities)

        for i in range(len(biluo_labels)):
            token = doc[i]
            label = biluo_labels[i]

            if label == '-':
                print('\nFOUND - at token ' + str(token.text))
                entities = self.fix_entities(entities, i)
                return self.get_labels(entities)

        biluo_simple = []

        for label in biluo_labels:
            if label != 'O':
                label = label[2:]
            biluo_simple.append(label)

        return biluo_simple