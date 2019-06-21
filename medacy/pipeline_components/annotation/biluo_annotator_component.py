import logging
from spacy.gold import biluo_tags_from_offsets
from spacy.tokens import Token
from ..base import BaseComponent
from medacy.tools import Annotations

class BiluoAnnotatorComponent(BaseComponent):
    """
    A pipeline component that overlays gold annotations. This pipeline component
    sets the attribute 'gold_label' to all tokens to be used as the class value of the token
    when fed into a supervised learning algorithm.

    """
    name = "gold_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline, labels):
        """
        :param spacy_pipeline: An exisiting spacy Language processing pipeline
        :param labels: The subset of labels from the gold annotations to restrict labeling to.
        """
        self.nlp = spacy_pipeline
        Token.set_extension('gold_label', default="O", force=True)

    def fix_entity(self, doc, entities, index):
        fixed_entity = False

        token = doc[index]
        token_start = token.idx
        token_end = token.idx + len(token.text)

        for i, entity in enumerate(entities):
            # If entity start cuts off token, shift start to beginning of token
            if entity[0] > token_start and entity[0] <= token_end:
                entity = (token_start, entity[1], entity[2])
                fixed_entity = True

            # If entity end cuts off token, shift end to end of token
            if entity[1] >= token_start and entity[1] < token_end:
                entity = (entity[0], token_end, entity[2])
                fixed_entity = True

            entities[i] = entity

        return entities

    def fix_entities(self, doc, entities, tags):
        broken_indices = []

        # Fix all of the entities by looking for '-' and expanding spans
        for i, tag in enumerate(tags):
            token = doc[i]

            if tag == '-':
                broken_indices.append(i)
                logging.debug('Found issue at (%d, %d, %s)' % (
                    token.idx,
                    token.idx + len(token),
                    token.text
                ))

        for i in broken_indices:
            tag = tags[i]

            if tag == '-':
                entities = self.fix_entity(doc, entities, i)

        return entities


    def __call__(self, doc):
        # Check that annnotations file is set
        if not hasattr(doc._, 'gold_annotation_file'):
            raise ValueError("No extension doc._.gold_annotation_file is present.")

        # Get annotations and convert to entities (offsets for spacy)
        gold_annotations = Annotations(doc._.gold_annotation_file, annotation_type='ann')
        
        entities = []
        for label, start, end, _ in gold_annotations.get_entity_annotations():
            entities.append((start, end, label))

        # Get biluo labels using spacy
        biluo_labels = biluo_tags_from_offsets(doc, entities)

        entities = self.fix_entities(doc, entities, biluo_labels)

        # Entities should be fixed now so get labels again
        biluo_labels = biluo_tags_from_offsets(doc, entities)

        # Simplify labels
        biluo_simple = []

        for label in biluo_labels:
            if label != 'O':
                label = label[2:]
            biluo_simple.append(label)

        # Set new gold labels in doc
        for token, label in zip(doc, biluo_simple):
            token._.set('gold_label', label)

        return doc