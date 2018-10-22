

class FeatureExtractor:
    """
    Extracting training data for use in a CRF.
    Features are given as rich dictionaries as described in:
    https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#features

    sklearn CRF suite is a wrapper for CRF suite that gives it a sci-kit compatability.
    """

    name = "feature_extractor"


    def __init__(self, window_size=5):
        #TODO this should have options for window_wize, features to exclude, and anything else.
        self.window_size = window_size
        pass

    def __call__(self, doc):

        features = [self._sent_to_feature_dicts(sent) for sent in doc.sents]
        labels = [self._sent_to_labels(sent) for sent in doc.sents]

        return (features, labels)

    def get_features_with_span_indices(self, doc):

        features = [self._sent_to_feature_dicts(sent) for sent in doc.sents]

        indices = [[(token.idx, token.idx+len(token)) for token in sent] for sent in doc.sents]

        return (features, indices)



    def _sent_to_feature_dicts(self, sent):
        return [self._token_to_feature_dict(i, sent) for i in range(len(sent))]

    def _sent_to_labels(self, sent, attribute='gold_label'):
        return [token._.get(attribute) for token in sent]

    def mapper_for_crf_wrapper(self, text):
        """
        CRF wrapper uses regexes to extract the output of the underlying C++ code.
        The inclusion of \n and space characters mess up these regexes, hence we map them to text here.
        :return:
        """
        if text == r"\n":
            return "#NEWLINE"
        if text == r"\t":
            return "#TAB"
        if text == " ":
            return "#SPACE"
        if text == "":
            return "#EMPTY"

        return text


    def _token_to_feature_dict(self, index, sentence):
        """

        :param index: the index of the token in the sentence
        :param sentence: an array of tokens corresponding to a sentence
        :return:
        """

        #This should automatically gather features that are set on tokens
        #by looping over all attributes set on sentence[index] that beging with 'feature'

        raise NotImplementedError();

        features = {
            'bias': 1.0,
            '0:text': self.mapper_for_crf_wrapper(sentence[index].text),
            '0:space_count': sentence[index].text.count(" "),
            '0:suffix': self.mapper_for_crf_wrapper(sentence[index].text[-3:]),
            '0:prefix': self.mapper_for_crf_wrapper(sentence[index].text[:3]),
            '0:pos': str(sentence[index].pos_),
            '0:cui': str(sentence[index]._.cui),
            '0:shape': str(sentence[index].shape),

            '0:phsu': sentence[index]._.is_phsu,
            '0:orch': sentence[index]._.is_orch,
            '0:sosy': sentence[index]._.is_sosy,
            '0:dsyn': sentence[index]._.is_dsyn,
            '0:fndg': sentence[index]._.is_fndg,
            '0:patf': sentence[index]._.is_patf,
            '0:aapp': sentence[index]._.is_aapp,
            '0:antb': sentence[index]._.is_antb,
            '0:inch': sentence[index]._.is_inch,
            '0:bacs': sentence[index]._.is_bacs,
            '0:qlco': sentence[index]._.is_qlco,
            '0:patf': sentence[index]._.is_patf,


            '0:is_form': sentence[index]._.is_form_unit,
            '0:is_route_type': sentence[index]._.is_route_type,
            '0:is_volume_unit': sentence[index]._.is_volume_unit,
            '0:is_measurement': sentence[index]._.is_measurement,
            '0:is_adr_lexicon_entry': sentence[index]._.is_adr_lexicon_entry,
            '0:is_frequency_indicator': sentence[index]._.is_frequency_indicator,
            '0:is_duration_pattern': sentence[index]._.is_duration_pattern
        }

        # special cases to insure access of window are not outside sentence bound.

        # if index > 0:
        #     features.update({
        #         '-1:text': self.mapper_for_crf_wrapper(sentence[index-1].text),
        #         '-1:space_count': sentence[index-1].text.count(" "),
        #         '-1:suffix': self.mapper_for_crf_wrapper(sentence[index-1].text[-3:]),
        #         '-1:prefix': self.mapper_for_crf_wrapper(sentence[index-1].text[:3]),
        #         '-1:pos': str(sentence[index-1].pos_),
        #         '-1:cui': str(sentence[index-1]._.cui),
        #         '-1:shape': str(sentence[index-1].shape),
        #
        #         '-1:phsu': sentence[index-1]._.is_phsu,
        #         '-1:orch': sentence[index-1]._.is_orch,
        #         '-1:sosy': sentence[index-1]._.is_sosy,
        #         '-1:dsyn': sentence[index-1]._.is_dsyn,
        #         '-1:fndg': sentence[index-1]._.is_fndg,
        #         '-1:patf': sentence[index-1]._.is_patf,
        #         '-1:aapp': sentence[index-1]._.is_aapp,
        #         '-1:antb': sentence[index-1]._.is_antb,
        #         '-1:inch': sentence[index-1]._.is_inch,
        #         '-1:bacs': sentence[index-1]._.is_bacs,
        #         '-1:qlco': sentence[index-1]._.is_qlco,
        #         '-1:patf': sentence[index-1]._.is_patf,
        #
        #         '-1:is_route_type': sentence[index -1]._.is_route_type,
        #         '-1:is_form': sentence[index-1]._.is_form_unit,
        #         '-1:is_volume_unit': sentence[index-1]._.is_volume_unit,
        #         '-1:is_measurement': sentence[index-1]._.is_measurement,
        #         '-1:is_adr_lexicon_entry': sentence[index-1]._.is_adr_lexicon_entry,
        #         '-1:is_frequency_indicator': sentence[index-1]._.is_frequency_indicator,
        #        '-1:is_duration_pattern': sentence[index-1]._.is_duration_pattern
        #
        #     })
        # else:
        #     features['BOS'] = True
        #
        # if index > 1:
        #     features.update({
        #         '-2:text': self.mapper_for_crf_wrapper(sentence[index - 2].text),
        #         '-2:space_count': sentence[index - 2].text.count(" "),
        #         '-2:pos': str(sentence[index - 2].pos_),
        #         '-2:suffix': self.mapper_for_crf_wrapper(sentence[index - 2].text[-3:]),
        #         '-2:prefix': self.mapper_for_crf_wrapper(sentence[index - 2].text[:3]),
        #         '-2:cui': str(sentence[index - 2]._.cui),
        #         '-2:shape': str(sentence[index - 2].shape),
        #
        #         '-2:phsu': sentence[index - 2]._.is_phsu,
        #         '-2:orch': sentence[index - 2]._.is_orch,
        #         '-2:sosy': sentence[index-2]._.is_sosy,
        #         '-2:dsyn': sentence[index-2]._.is_dsyn,
        #         '-2:fndg': sentence[index-2]._.is_fndg,
        #         '-2:patf': sentence[index-2]._.is_patf,
        #         '-2:aapp': sentence[index-2]._.is_aapp,
        #         '-2:antb': sentence[index-2]._.is_antb,
        #         '-2:inch': sentence[index-2]._.is_inch,
        #         '-2:bacs': sentence[index-2]._.is_bacs,
        #         '-2:qlco': sentence[index-2]._.is_qlco,
        #         '-2:patf': sentence[index-2]._.is_patf,
        #
        #         '-2:is_route_type': sentence[index - 2]._.is_route_type,
        #         '-2:is_form': sentence[index - 2]._.is_form_unit,
        #         '-2:is_volume_unit': sentence[index - 2]._.is_volume_unit,
        #         '-2:is_measurement': sentence[index - 2]._.is_measurement,
        #         '-2:is_adr_lexicon_entry': sentence[index - 2]._.is_adr_lexicon_entry,
        #         '-2:is_frequency_indicator': sentence[index - 2]._.is_frequency_indicator,
        #        '-2:is_duration_pattern': sentence[index -2]._.is_duration_pattern
        #     })
        #
        # if index > 2:
        #     features.update({
        #         '-3:text': self.mapper_for_crf_wrapper(sentence[index - 3].text),
        #         '-3:space_count': sentence[index - 3].text.count(" "),
        #         '-3:pos': str(sentence[index - 3].pos_),
        #         '-3:suffix': self.mapper_for_crf_wrapper(sentence[index - 3].text[-3:]),
        #         '-3:prefix': self.mapper_for_crf_wrapper(sentence[index - 3].text[:3]),
        #         '-3:cui': str(sentence[index - 3]._.cui),
        #         '-3:shape': str(sentence[index - 3].shape),
        #
        #         '-3:phsu': sentence[index - 3]._.is_phsu,
        #         '-3:orch': sentence[index - 3]._.is_orch,
        #         '-3:sosy': sentence[index - 3]._.is_sosy,
        #         '-3:dsyn': sentence[index - 3]._.is_dsyn,
        #         '-3:fndg': sentence[index - 3]._.is_fndg,
        #         '-3:patf': sentence[index - 3]._.is_patf,
        #         '-3:aapp': sentence[index - 3]._.is_aapp,
        #         '-3:antb': sentence[index - 3]._.is_antb,
        #         '-3:inch': sentence[index - 3]._.is_inch,
        #         '-3:bacs': sentence[index - 3]._.is_bacs,
        #         '-3:qlco': sentence[index - 3]._.is_qlco,
        #         '-3:patf': sentence[index - 3]._.is_patf,
        #
        #         '-3:is_route_type': sentence[index - 3]._.is_route_type,
        #         '-3:is_form': sentence[index - 3]._.is_form_unit,
        #         '-3:is_volume_unit': sentence[index - 3]._.is_volume_unit,
        #         '-3:is_measurement': sentence[index - 3]._.is_measurement,
        #         '-3:is_adr_lexicon_entry': sentence[index - 3]._.is_adr_lexicon_entry,
        #         '-3:is_frequency_indicator': sentence[index - 3]._.is_frequency_indicator,
        #         '-3:is_duration_pattern': sentence[index - 3]._.is_duration_pattern
        #     })
        #
        # if index < len(sentence)-1:
        #     features.update({
        #         '+1:text': self.mapper_for_crf_wrapper(sentence[index+1].text),
        #         '+1:space_count': sentence[index+1].text.count(" "),
        #         '+1:pos': str(sentence[index+1].pos_),
        #         '+1:suffix': self.mapper_for_crf_wrapper(sentence[index + 1].text[-3:]),
        #         '+1:prefix': self.mapper_for_crf_wrapper(sentence[index + 1].text[:3]),
        #         '+1:cui': str(sentence[index+1]._.cui),
        #         '+1:shape': str(sentence[index+1].shape),
        #
        #         '+1:phsu': sentence[index+1]._.is_phsu,
        #         '+1:orch': sentence[index + 1]._.is_orch,
        #         '+1:sosy': sentence[index+1]._.is_sosy,
        #         '+1:dsyn': sentence[index+1]._.is_dsyn,
        #         '+1:fndg': sentence[index+1]._.is_fndg,
        #         '+1:patf': sentence[index+1]._.is_patf,
        #         '+1:aapp': sentence[index+1]._.is_aapp,
        #         '+1:antb': sentence[index+1]._.is_antb,
        #         '+1:inch': sentence[index+1]._.is_inch,
        #         '+1:bacs': sentence[index+1]._.is_bacs,
        #         '+1:qlco': sentence[index+1]._.is_qlco,
        #         '+1:patf': sentence[index+1]._.is_patf,
        #
        #         '+1:is_route_type': sentence[index + 1]._.is_route_type,
        #         '+1:is_form': sentence[index + 1]._.is_form_unit,
        #         '+1:is_volume_unit': sentence[index + 1]._.is_volume_unit,
        #         '+1:is_measurement': sentence[index+1]._.is_measurement,
        #         '+1:is_adr_lexicon_entry': sentence[index+1]._.is_adr_lexicon_entry,
        #         '+1:is_frequency_indicator': sentence[index+1]._.is_frequency_indicator,
        #         '+1:is_duration_pattern': sentence[index+1]._.is_duration_pattern
        #     })
        # else:
        #     features['EOS'] = True
        #
        # if index < len(sentence)-2:
        #     features.update({
        #         '+2:text': self.mapper_for_crf_wrapper(sentence[index+2].text),
        #         '+2:space_count': sentence[index+2].text.count(" "),
        #         '+2:suffix': self.mapper_for_crf_wrapper(sentence[index +2].text[-3:]),
        #         '+2:prefix': self.mapper_for_crf_wrapper(sentence[index + 2].text[:3]),
        #         '+2:pos': str(sentence[index+2].pos_),
        #         '+2:cui': str(sentence[index+2]._.cui),
        #         '+2:shape': str(sentence[index+2].shape),
        #
        #         '+2:phsu': sentence[index+2]._.is_phsu,
        #         '+2:orch': sentence[index+2]._.is_orch,
        #         '+2:sosy': sentence[index+2]._.is_sosy,
        #         '+2:dsyn': sentence[index+2]._.is_dsyn,
        #         '+2:fndg': sentence[index+2]._.is_fndg,
        #         '+2:patf': sentence[index+2]._.is_patf,
        #         '+2:aapp': sentence[index+2]._.is_aapp,
        #         '+2:antb': sentence[index+2]._.is_antb,
        #         '+2:inch': sentence[index+2]._.is_inch,
        #         '+2:bacs': sentence[index+2]._.is_bacs,
        #         '+2:qlco': sentence[index+2]._.is_qlco,
        #         '+2:patf': sentence[index+2]._.is_patf,
        #
        #         '+2:is_route_type': sentence[index + 2]._.is_route_type,
        #         '+2:is_form': sentence[index + 2]._.is_form_unit,
        #         '+2:is_volume_unit': sentence[index + 2]._.is_volume_unit,
        #         '+2:is_measurement': sentence[index+2]._.is_measurement,
        #         '+2:is_adr_lexicon_entry': sentence[index+2]._.is_adr_lexicon_entry,
        #         '+2:is_frequency_indicator': sentence[index + 2]._.is_frequency_indicator,
        #         '+2:is_duration_pattern': sentence[index+2]._.is_duration_pattern
        #     })
        #
        # if index < len(sentence) - 3:
        #     features.update({
        #         '+3:text': self.mapper_for_crf_wrapper(sentence[index + 3].text),
        #         '+3:space_count': sentence[index + 3].text.count(" "),
        #         '+3:suffix': self.mapper_for_crf_wrapper(sentence[index + 3].text[-3:]),
        #         '+3:prefix': self.mapper_for_crf_wrapper(sentence[index + 3].text[:3]),
        #         '+3:pos': str(sentence[index + 3].pos_),
        #         '+3:cui': str(sentence[index + 3]._.cui),
        #         '+3:shape': str(sentence[index + 3].shape),
        #
        #         '+3:phsu': sentence[index + 3]._.is_phsu,
        #         '+3:orch': sentence[index + 3]._.is_orch,
        #         '+3:sosy': sentence[index + 3]._.is_sosy,
        #         '+3:dsyn': sentence[index + 3]._.is_dsyn,
        #         '+3:fndg': sentence[index + 3]._.is_fndg,
        #         '+3:patf': sentence[index + 3]._.is_patf,
        #         '+3:aapp': sentence[index + 3]._.is_aapp,
        #         '+3:antb': sentence[index + 3]._.is_antb,
        #         '+3:inch': sentence[index + 3]._.is_inch,
        #         '+3:bacs': sentence[index + 3]._.is_bacs,
        #         '+3:qlco': sentence[index + 3]._.is_qlco,
        #         '+3:patf': sentence[index + 3]._.is_patf,
        #
        #         '+3:is_route_type': sentence[index + 3]._.is_route_type,
        #         '+3:is_form': sentence[index + 3]._.is_form_unit,
        #         '+3:is_volume_unit': sentence[index + 3]._.is_volume_unit,
        #         '+3:is_measurement': sentence[index + 3]._.is_measurement,
        #         '+3:is_adr_lexicon_entry': sentence[index + 3]._.is_adr_lexicon_entry,
        #         '+3:is_frequency_indicator': sentence[index + 3]._.is_frequency_indicator,
        #         '+3:is_duration_pattern': sentence[index + 3]._.is_duration_pattern
        #     })


        return features





