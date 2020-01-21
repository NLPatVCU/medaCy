import logging

from spacy.tokens import Token

from medacy.data.annotations import Annotations
from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer


class GoldAnnotatorOverlayer(BaseOverlayer):
    """
    A pipeline component that overlays gold annotations. This pipeline component sets the attribute 'gold_label'
    to all tokens to be used as the class value of the token when fed into a supervised learning algorithm.
    Note that these annotations are not used as features.
    """

    name = "gold_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline, labels):
        """
        :param spacy_pipeline: An existing spaCy pipeline
        :param labels: The subset of labels from the gold annotations to restrict labeling to.
        """
        super().__init__(
            component_name=self.name,
            dependencies=self.dependencies
        )
        self.nlp = spacy_pipeline
        self.labels = labels
        self.failed_overlay_count = 0
        self.failed_identifying_span_count = 0
        Token.set_extension('gold_label', default="O", force=True)

    def find_span(self, start, end, doc):
        """
        Greedily searches characters around word to find a valid set of tokens the annotation likely corresponds to.
        :param start: index of token start
        :param end: index of token end
        :param doc: spaCy Doc object
        :return:
        """
        greedy_searched_span = doc.char_span(start, end)
        if greedy_searched_span is not None:
            return greedy_searched_span

        greedy_searched_span = doc.char_span(start, end-1) #annotation may have extended over an ending blank space
        if greedy_searched_span is not None:
            return greedy_searched_span

        # No clue - increase boundaries incrementally until a valid span is found.
        i = 0
        while greedy_searched_span is None and i <= 20:
            if i % 2 == 0:
                end += 1
            else:
                start -= 1
            i += 1
            greedy_searched_span = doc.char_span(start, end)

        return greedy_searched_span

    def __call__(self, doc):
        """
        Overlays entity annotations over tokens in a Doc object. Requires that tokens in the Doc have the custom
        'gold_annotation_file' and 'file_name' extension.
        :param doc: a spaCy Doc object.
        :return: the same Doc object, but it now has 'gold_label' annotations.
        """

        logging.debug("%s: Called GoldAnnotator Component", doc._.file_name)

        # check if gold annotation file path has been set.
        if not hasattr(doc._, 'gold_annotation_file'):
            logging.warning("No extension doc._.gold_annotation_file is present; it will not be possible to fit a model with this Doc")
            return doc

        gold_annotations = Annotations(doc._.gold_annotation_file)

        for e_label, e_start, e_end, _ in gold_annotations.get_entity_annotations():
            if e_start > e_end:
                logging.critical("%s: Broken annotation - start is greater than end: (%i,%i,%s)",
                                 doc._.file_name, e_start, e_end, e_label)
                continue
            span = doc.char_span(e_start, e_end)

            if span is None:
                self.failed_overlay_count += 1
                self.failed_identifying_span_count += 1
                logging.warning("%s: Number of failed annotation overlays with current tokenizer: %i (%i,%i,%s)",
                                doc._.file_name, self.failed_overlay_count, e_start, e_end, e_label)

            fixed_span = self.find_span(e_start, e_end, doc)
            if fixed_span is not None:
                if span is None:
                    logging.warning("%s: Fixed span (%i,%i,%s) into: %s",
                                    doc._.file_name, e_start, e_end, e_label, fixed_span.text)
                    self.failed_identifying_span_count -= 1
                for token in fixed_span:
                    if e_label in self.labels or not self.labels:
                        token._.set('gold_label', e_label)

            else:  # annotation was not able to be fixed, it will be ignored - this is bad in evaluation.
                logging.warning("%s: Could not fix annotation: (%i,%i,%s)", doc._.file_name, e_start, e_end, e_label)
                logging.warning("%s: Total Failed Annotations: %i", doc._.file_name, self.failed_identifying_span_count)

        if self.failed_overlay_count > .3 * len(gold_annotations):
            logging.warning("%s: Annotations may mis-aligned as more than 30 percent failed to overlay: %s",
                            doc._.file_name, doc._.gold_annotation_file)

        return doc
