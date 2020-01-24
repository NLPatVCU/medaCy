from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer


class VolumeUnitOverlayer(BaseOverlayer):
    """
    A pipeline component that tags volume units
    """

    name="volume_unit_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline):
        self.nlp = spacy_pipeline
        Token.set_extension('feature_is_volume_unit', default=False)
        self.nlp.entity.add_label('volume_unit')
        self.volume_matcher = Matcher(self.nlp.vocab)

        self.volume_matcher.add('UNIT_OF_VOLUME', None,
                                [{'LOWER': 'ml'}],
                                 [{'ORTH': 'dL'}],
                                [{'LOWER': 'cc'}],
                                [{'ORTH': 'L'}])

    def __call__(self, doc):
        nlp = self.nlp
        with doc.retokenize() as retokenizer:
            #match and tag volume units
            matches = self.volume_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['volume_unit'])
                for token in span:
                    token._.feature_is_volume_unit = True
                if len(span) > 1:
                    retokenizer.merge(span)
                doc.ents = list(doc.ents) + [span]
        return doc