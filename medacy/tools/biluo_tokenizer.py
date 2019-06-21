import logging
import spacy
from spacy.gold import biluo_tags_from_offsets
from spacy.gold import offsets_from_biluo_tags

class BiluoTokenizer:
    doc = None

    def __init__(self, doc):
        self.doc = doc

    def get_tokens(self):
        return [str(token) for token in self.doc]

    def fix_entities(self, entities, index):
        fixed_entity = False

        doc = self.doc
        token = doc[index]
        start = token.idx
        end = token.idx + len(token.text)

        for i, entity in enumerate(entities):
            # If entity start cuts off token, shift start to beginning of token
            if entity[0] > start and entity[0] <= end:
                entity = (start, entity[1], entity[2])
                fixed_entity = True

            # If entity end cuts off token, shift end to end of token
            if entity[1] >= start and entity[1] < end:
                entity = (entity[0], end, entity[2])
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
                logging.debug('Found issue at (%d, %d, %s)' % (
                    token.idx,
                    token.idx + len(token),
                    token.text
                ))
                entities = self.fix_entities(entities, i)
                return self.get_labels(entities)

        biluo_simple = []

        for label in biluo_labels:
            if label != 'O':
                label = label[2:]
            biluo_simple.append(label)

        return biluo_simple

    def get_entities(self, tags):
        doc = self.doc

        b_indexes = []
        i_indexes = []
        l_indexes = []
        u_indexes = []

        for i, tag in enumerate(tags):
            if tag != 'O':
                if i == 0:
                    if tags[i+1] == 'O':
                        u_indexes.append(i)
                    else:
                        b_indexes.append(i)
                elif i == len(tags) - 1:
                    if tags[i-1] == 'O':
                        u_indexes.append(i)
                    else:
                        l_indexes.append(i)
                elif tags[i-1] == 'O' and tags[i+1] == 'O':
                    u_indexes.append(i)
                elif tags[i-1] == 'O':
                    b_indexes.append(i)
                elif tags[i+1] == 'O':
                    l_indexes.append(i)
                else:
                    i_indexes.append(i)
        
        action_dictionary = {
            'B-': b_indexes,
            'I-': i_indexes,
            'L-': l_indexes,
            'I-': i_indexes,
            'U-': u_indexes
        }

        for action, indexes in action_dictionary.items():
            for index in indexes:
                tags[index] = action + tags[index]

        entities = offsets_from_biluo_tags(doc, tags)

        return entities
