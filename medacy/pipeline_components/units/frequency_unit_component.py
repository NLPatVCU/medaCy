from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer


class FrequencyUnitOverlayer(BaseOverlayer):
    """
    A pipeline component that tags Frequency units
    """

    name="frequency_unit_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline):
        self.nlp = spacy_pipeline
        Token.set_extension('feature_is_frequency_indicator', default=False)
        self.nlp.entity.add_label('frequency_indicator')
        self.frequency_matcher = Matcher(self.nlp.vocab)

        self.frequency_matcher.add('FREQUENCY_MATCHER', None,
                               [{'LOWER': 'bid'}],
                               [{'LOWER': 'prn'}],
                               [{'LOWER': 'qid'}],
                               [{'LOWER': 'tid'}],
                               [{'LOWER': 'qd'}],
                               [{'LOWER': 'daily'}],
                               [{'LOWER': 'hs'}],
                               [{'LOWER': 'as'}, {'LOWER': 'needed'}],
                               [{'LOWER': 'once'}, {'LOWER': 'a'}, {'LOWER': 'day'}],
                               [{'LOWER': 'twice'}, {'LOWER': 'a'}, {'LOWER': 'day'}]
                               )

    def __call__(self, doc):
        nlp = self.nlp
        with doc.retokenize() as retokenizer:
            # match and frequency indicators
            matches = self.frequency_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['frequency_indicator'])
                for token in span:
                    token._.feature_is_frequency_indicator = True
                if len(span) > 1:
                    retokenizer.merge(span)
                doc.ents = list(doc.ents) + [span]
        return doc