
from spacy.tokens import Token
from spacy.matcher import Matcher
from spacy.tokens import Span
from medacy.pipeline_components.base_components import BaseOverlayer
from ...pipeline_components import MassUnitComponent, TimeUnitComponent, VolumeUnitComponent

class MeasurementUnitComponent(BaseOverlayer):
    """
    A pipeline component that tags Frequency units
    """

    name="measurement_unit_annotator"
    dependencies = [MassUnitComponent, TimeUnitComponent, VolumeUnitComponent]

    def __init__(self, spacy_pipeline):
        self.nlp = spacy_pipeline
        Token.set_extension('feature_is_measurement_unit', default=False)
        self.nlp.entity.add_label('measurement_unit')
        self.unit_of_measurement_matcher = Matcher(self.nlp.vocab)

        self.unit_of_measurement_matcher.add('UNIT_OF_MEASUREMENT', None,
                         [{'ENT_TYPE': 'mass_unit'}, {'ORTH': '/'}, {'ENT_TYPE': 'volume_unit'}],
                         [{'ENT_TYPE': 'volume_unit'}, {'ORTH': '/'}, {'ENT_TYPE': 'time_unit'}],
                         [{'ENT_TYPE': 'form_unit'}, {'ORTH': '/'}, {'ENT_TYPE': 'volume_unit'}]
                         )

    def __call__(self, doc):
        nlp = self.nlp
        with doc.retokenize() as retokenizer:
            # match units of measurement (x/y, , etc)
            matches = self.unit_of_measurement_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['measurement_unit'])
                for token in span:
                    token._.feature_is_measurement_unit = True
                if len(span) > 1:
                    retokenizer.merge(span)
                doc.ents = list(doc.ents) + [span]
        return doc