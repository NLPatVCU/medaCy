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
        Token.set_extension('gold_label', default='O', force=True)

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

        greedy_searched_span = doc.char_span(start, end - 1)  # annotation may have extended over an ending blank space
        if greedy_searched_span is not None:
            return greedy_searched_span

        # No clue - increase boundaries incrementally until a valid span is found.
        i = 0
        while greedy_searched_span is None and i <= 20:
            end += 1 if i % 2 == 0 else -1
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

        file_name = doc._.file_name
        logging.debug(f"{file_name}: Called GoldAnnotator Component")

        failed_overlay_count = 0
        failed_identifying_span_count = 0

        # check if gold annotation file path has been set.
        if not hasattr(doc._, 'gold_annotation_file'):
            logging.warning(f"doc._.gold_annotation_file not defined for {file_name}; "
                            f"it will not be possible to fit a model with this Doc")
            return doc

        gold_annotations = Annotations(doc._.gold_annotation_file)

        for ent in gold_annotations:
            if ent.start > ent.end:
                logging.critical(f"{file_name}: Broken annotation - start is greater than end: {ent}")
                continue

            span = doc.char_span(ent.start, ent.end)

            if span is None:
                failed_overlay_count += 1
                failed_identifying_span_count += 1

            fixed_span = self.find_span(ent.start, ent.end, doc)
            if fixed_span is not None:
                if span is None:
                    logging.warning(f"{file_name}: Fixed {ent} into: {fixed_span.text}")
                    failed_identifying_span_count -= 1

                for token in fixed_span:
                    if ent.tag in self.labels or not self.labels:
                        token._.set('gold_label', ent.tag)

            else:
                # Annotation was not able to be fixed, it will be ignored - this is bad in evaluation.
                logging.warning(f"{file_name}: Could not fix annotation: {ent}")

        logging.warning(f"{file_name}: Number of failed annotation overlays with current tokenizer: {failed_overlay_count}")

        if failed_overlay_count > .3 * len(gold_annotations):
            logging.critical(f"{file_name}: More than 30% of annotations failed to overlay")

        return doc
