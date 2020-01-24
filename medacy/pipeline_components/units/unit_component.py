import logging

from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer


class UnitOverlayer(BaseOverlayer):
    """
    A pipeline component that tags units.
    Begins by first tagging all mass, volume, time, and form units then aggregates as necessary.
    """

    name="unit_annotator"
    dependencies = []

    def __init__(self, nlp):
        self.nlp = nlp
        Token.set_extension('feature_is_mass_unit', default=False, force=True)
        nlp.entity.add_label('mass_unit')

        Token.set_extension('feature_is_volume_unit', default=False, force=True)
        nlp.entity.add_label('volume_unit')

        Token.set_extension('feature_is_time_unit', default=False, force=True)
        nlp.entity.add_label('time_unit')

        Token.set_extension('feature_is_route_type', default=False, force=True)
        nlp.entity.add_label('route_type')

        Token.set_extension('feature_is_form_unit', default=False, force=True)
        nlp.entity.add_label('form_unit')

        Token.set_extension('feature_is_frequency_indicator', default=False, force=True)
        nlp.entity.add_label('frequency_indicator')


        Token.set_extension('feature_is_measurement_unit', default=False, force=True)
        nlp.entity.add_label('measurement_unit')

        Token.set_extension('feature_is_measurement', default=False, force=True)
        nlp.entity.add_label('measurement')

        Token.set_extension('feature_is_duration_pattern', default=False)
        nlp.entity.add_label('duration_pattern')



        self.mass_matcher = Matcher(nlp.vocab)
        self.volume_matcher = Matcher(nlp.vocab)
        self.time_matcher = Matcher(nlp.vocab)
        self.route_matcher = Matcher(nlp.vocab)
        self.form_matcher = Matcher(nlp.vocab)
        self.unit_of_measurement_matcher = Matcher(nlp.vocab)
        self.measurement_matcher = Matcher(nlp.vocab)
        self.frequency_matcher = Matcher(nlp.vocab)
        self.duration_matcher = Matcher(nlp.vocab)

        self.mass_matcher.add('UNIT_OF_MASS', None,
                              [{'LOWER': 'mcg'}],
                              [{'LOWER': 'microgram'}],
                              [{'LOWER': 'micrograms'}],
                              [{'ORTH': 'mg'}],
                              [{'LOWER': 'milligram'}],
                              [{'LOWER': 'g'}],
                              [{'LOWER': 'kg'}],
                              [{'ORTH': 'mEq'}])

        self.volume_matcher.add('UNIT_OF_VOLUME', None,
                                [{'LOWER': 'ml'}],
                                 [{'ORTH': 'dL'}],
                                [{'LOWER': 'cc'}],
                                [{'ORTH': 'L'}])

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


        self.form_matcher.add('UNIT_OF_FORM', None,
                              [{'ORTH': 'dose'}],
                              [{'ORTH': 'doses'}],
                              [{'LEMMA': 'pill'}],
                              [{'LEMMA': 'tablet'}],
                              [{'LEMMA': 'unit'}],
                              [{'LEMMA': 'u'}],
                              [{'LEMMA': 'patch'}],
                              [{'LEMMA': 'unit'}],
                              [{'ORTH': 'lotion'}],
                              [{'ORTH': 'powder'}],
                              [{'ORTH': 'amps'}],
                              [{'LOWER': 'actuation'}],
                              [{'LEMMA': 'suspension'}],
                              [{'LEMMA': 'syringe'}],
                              [{'LEMMA': 'puff'}],
                              [{'LEMMA': 'liquid'}],
                              [{'LEMMA': 'aerosol'}],
                              [{'LEMMA': 'cap'}]
                              )

        self.route_matcher.add('TYPE_OF_ROUTE', None,
                               [{'LOWER': 'IV'}],
                               [{'ORTH': 'intravenous'}],
                               [{'LOWER': 'po'}],
                               [{'ORTH': 'gtt'}],
                               [{'LOWER': 'drip'}],
                               [{'LOWER': 'inhalation'}],
                               [{'LOWER': 'by'}, {'LOWER': 'mouth'}],
                               [{'LOWER': 'topical'}],
                               [{'LOWER': 'subcutaneous'}],
                               [{'LOWER': 'ophthalmic'}],
                               [{'LEMMA': 'injection'}],
                               [{'LOWER': 'mucous'}, {'LOWER': 'membrane'}],
                               [{'LOWER': 'oral'}],
                               [{'LOWER': 'nebs'}],
                               [{'LOWER': 'transdermal'}],
                               [{'LOWER': 'nasal'}]
                              )


        self.unit_of_measurement_matcher.add('UNIT_OF_MEASUREMENT', None,
                         [{'ENT_TYPE': 'mass_unit'}, {'ORTH': '/'}, {'ENT_TYPE': 'volume_unit'}],
                         [{'ENT_TYPE': 'volume_unit'}, {'ORTH': '/'}, {'ENT_TYPE': 'time_unit'}],
                         [{'ENT_TYPE': 'form_unit'}, {'ORTH': '/'}, {'ENT_TYPE': 'volume_unit'}]
                         )
        self.measurement_matcher.add('MEASUREMENT', None,
                         [{'LIKE_NUM': True}, {'ORTH': '%'}],
                         [{'LIKE_NUM': True}, {'ENT_TYPE': 'measurement_unit'}],
                         [{'LIKE_NUM': True}, {'ENT_TYPE': 'mass_unit'}],
                         [{'LIKE_NUM': True}, {'ENT_TYPE': 'volume_unit'}],
                         [{'LIKE_NUM': True}, {'ENT_TYPE': 'form_unit'}],
                         [{'LIKE_NUM': True},{'LOWER': 'x'}, {'ENT_TYPE': 'form_unit'}]

                         )

        self.duration_matcher.add('DURATION', None,
                                  [{'POS': 'PREP'}, {'LIKE_NUM': True}, {'ENT_TYPE': 'time_unit'}],
                                  [{'LIKE_NUM': True}, {'ENT_TYPE': 'time_unit'}],
                                  [{'LOWER': 'in'}, {'LIKE_NUM': True},{'ENT_TYPE': 'time_unit'}],
                                  [{'LOWER': 'prn'}]
                                  )


    def __call__(self, doc):
        logging.debug("Called UnitAnnotator Component")
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
                try:
                    if len(span) > 1:
                        retokenizer.merge(span)
                except ValueError:
                    pass
                doc.ents = list(doc.ents) + [span]

        with doc.retokenize() as retokenizer:
            #match and tag volume units
            matches = self.volume_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['volume_unit'])
                for token in span:
                    token._.feature_is_volume_unit = True
                try:
                    if len(span) > 1:
                        retokenizer.merge(span)
                except ValueError:
                    pass
                doc.ents = list(doc.ents) + [span]


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

        with doc.retokenize() as retokenizer:
            # durations
            matches = self.duration_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['duration_pattern'])
                for token in span:
                    token._.feature_is_duration_pattern = True
                try:
                    if len(span) > 1:
                        retokenizer.merge(span)
                except ValueError:
                    pass

                doc.ents = list(doc.ents) + [span]

        with doc.retokenize() as retokenizer:

            # match and frequency indicators
            matches = self.frequency_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['frequency_indicator'])
                for token in span:
                    token._.feature_is_frequency_indicator = True
                try:
                    if len(span) > 1:
                        retokenizer.merge(span)
                except ValueError:
                    pass
                doc.ents = list(doc.ents) + [span]

        with doc.retokenize() as retokenizer:
            #match and tag form units
            matches = self.form_matcher(doc)
            spans = []
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['form_unit'])
                for token in span:
                    token._.feature_is_form_unit = True
                try:
                    if len(span) > 1:
                        retokenizer.merge(span)
                except ValueError:
                    pass
                doc.ents = list(doc.ents) + [span]

        with doc.retokenize() as retokenizer:
            # match and tag route types
            matches = self.route_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['route_type'])
                for token in span:
                    token._.feature_is_route_type = True
                    try:
                        if len(span) > 1:
                            retokenizer.merge(span)
                    except ValueError:
                        pass
                    doc.ents = list(doc.ents) + [span]

        with doc.retokenize() as retokenizer:
            # match units of measurement (x/y, , etc)
            matches = self.unit_of_measurement_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['measurement_unit'])
                for token in span:
                    token._.feature_is_measurement_unit = True
                try:
                    if len(span) > 1:
                        retokenizer.merge(span)
                except ValueError:
                    pass
                doc.ents = list(doc.ents) + [span]

        with doc.retokenize() as retokenizer:

            # units of measures, numbers , percentages all together
            matches = self.measurement_matcher(doc)
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=nlp.vocab.strings['measurement'])
                for token in span:
                    token._.feature_is_measurement = True
                try:
                    if len(span) > 1:
                        retokenizer.merge(span)
                except ValueError:
                    pass
                doc.ents = list(doc.ents) + [span]

        return doc