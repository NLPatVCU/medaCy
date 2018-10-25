from ..base import BaseComponent
from medacy.tools.ann_to_json import ann_to_json
from spacy.tokens import Token
import logging

class GoldAnnotatorComponent(BaseComponent):
    #TODO CLEAN ME
    #TODO Look into spacy GoldParse
    #TODO This code really needs a fixing but it is bootstrapped to work from development
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
        self.labels = labels
        self.failed_overlay_count = 0
        self.failed_identifying_span_count = 0

    def find_span(self, start, end, label, span, doc):
        #TODO REALLY clean this up asap - this method will find valid spans with annotation boundaries that
        #TODO do not line up with tokenization boundaries.

        span1 = None
        span2 = None
        span_new = None
        if span is None:
            w = 1
            while span1 is None:
                s1 = start - w
                span1 = doc.char_span(s1, end)
                if span1 is not None:
                    span_new = span1
                    # span_sent = str(span.sent).split()
                if w == 10:
                    if span1 is None:
                        z = 1

                        while span2 is None:
                            e1 = end + z
                            span2 = doc.char_span(start, e1)
                            if span2 is not None:
                                span_new = span2
                                # span_sent = str(span.sent).split()
                            if z == 20:
                                if span2 is None:
                                    logging.warning("Could not overlay span: %s %s %s %s", str(start), str(end), str(label), str(span2))
                                break
                            z = z + 1

                    break
                w = w + 1
        else:
            span_new = span
        return span_new

    def __call__(self, doc):
        nlp = self.nlp
        logging.debug("Called GoldAnnotator Component")

        #check if gold annotation file path has been set.
        if not hasattr(doc._, 'gold_annotation_file'):
            raise ValueError("No extension doc._.gold_annotation_file is present.")

        with open(doc._.gold_annotation_file, 'r') as gold_annotations:
            gold_annotations = ann_to_json(gold_annotations.read())

        # for label in set([label for _,_,label in [gold['entities'][key] for key in gold['entities']]]):
        Token.set_extension('gold_label', default="O", force=True)
        # for token in doc:
        #     print(token.text, token.idx)
        for e_start, e_end, e_label in [gold_annotations['entities'][key] for key in gold_annotations['entities']]:

            span = doc.char_span(e_start, e_end)
            if span is None:
                self.failed_overlay_count += 1
                self.failed_identifying_span_count += 1
                logging.warning("Number of failed annotation overlays with current tokenizer: %i (%i,%i,%s)", self.failed_overlay_count, e_start, e_end, e_label)
            fixed_span = self.find_span(e_start, e_end, e_label, span, doc)
            if fixed_span is not None:
                if span is None:
                    logging.warning("Fixed span (%i,%i,%s) into: %s", e_start, e_end, e_label,fixed_span.text)
                    self.failed_identifying_span_count -= 1
                for token in fixed_span:
                    if e_label in self.labels or not self.labels:
                        token._.set('gold_label', e_label)

            else: #annotation was not able to be fixed, it will be ignored - this is bad in evaluation.
                logging.warning("Could not fix annotation: (%i,%i,%s)", e_start, e_end, e_label)
                logging.warning("Total Failed Annotations: %i", self.failed_identifying_span_count)

        return doc




