from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer


class TimeUnitOverlayer(BaseOverlayer):
    """
    A pipeline component that tags time units
    """

    name="time_unit_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline):
        self.nlp = spacy_pipeline
        Token.set_extension('feature_is_time_unit', default=False)
        self.nlp.entity.add_label('time_unit')
        self.time_matcher = Matcher(self.nlp.vocab)

        self.time_matcher.add('UNIT_OF_TIME', None,
                              [{'LOWER': 'sec'}],
                              [{'LOWER': 'second'}],
                              [{'LOWER': 'seconds'}],
                              [{'LOWER': 'min'}],
                              [{'LOWER': 'minute'}],
                              [{'LOWER': 'minutes'}],
                              [{'LOWER': 'hr'}],
                              [{'LOWER': 'hour'}],
                              [{'LOWER': 'day'}],
                              [{'LOWER': 'days'}],
                              [{'LOWER': 'week'}],
                              [{'LOWER': 'weeks'}],
                              [{'LOWER': 'month'}],
                              [{'LOWER': 'months'}],
                              [{'LOWER': 'year'}],
                              [{'LOWER': 'years'}],
                              [{'LOWER': 'yrs'}]
                              )

    def __call__(self, doc):
        nlp = self.nlp
        with doc.retokenize() as retokenizer:
            # match and tag time units
            matches = self.time_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['time_unit'])
                for token in span:
                    token._.feature_is_time_unit = True
                if len(span) > 1:
                    retokenizer.merge(span)
                doc.ents = list(doc.ents) + [span]
        return doc