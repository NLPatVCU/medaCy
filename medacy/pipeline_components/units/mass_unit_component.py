from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer


class MassUnitOverlayer(BaseOverlayer):
    """
    A pipeline component that tags mass units
    """

    name="mass_unit_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline):
        self.nlp = spacy_pipeline
        Token.set_extension('feature_is_mass_unit', default=False)
        self.nlp.entity.add_label('mass_unit')
        self.mass_matcher = Matcher(self.nlp.vocab)

        self.mass_matcher.add('UNIT_OF_MASS', None,
                              [{'LOWER': 'mcg'}],
                              [{'LOWER': 'microgram'}],
                              [{'LOWER': 'micrograms'}],
                              [{'ORTH': 'mg'}],
                              [{'LOWER': 'milligram'}],
                              [{'LOWER': 'g'}],
                              [{'LOWER': 'kg'}],
                              [{'ORTH': 'mEq'}])

    def __call__(self, doc):
        nlp = self.nlp
        with doc.retokenize() as retokenizer:
            #match and tag mass units
            matches = self.mass_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['mass_unit'])
                if span is None:
                    raise BaseException("Span is none")
                for token in span:
                    token._.feature_is_mass_unit = True
                if len(span) > 1:
                    retokenizer.merge(span)
                doc.ents = list(doc.ents) + [span]
        return doc