from ..base import BaseComponent
from medacy.tools.ann_to_json import ann_to_json
from spacy.tokens import Token

class GoldAnnotatorComponent(BaseComponent):
    #TODO CLEAN ME
    #TODO Look into spacy GoldParse
    """
    A pipeline component that overlays gold annotations. This pipeline component
    sets the attribute 'gold_label' to all tokens to be used as the class value of the token
    when fed into a supervised learning algorithm.

    """

    name = "gold_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline, labels = []):
        """
        :param spacy_pipeline: An exisiting spacy Language processing pipeline
        :param labels: The subset of labels from the gold annotations to restrict labeling to.
        """
        self.nlp = spacy_pipeline
        self.labels = labels
        self.none_count = 0

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
                                    print(start, end, label, span2)
                                break
                            z = z + 1

                    break
                w = w + 1
        else:
            span_new = span
        return span_new

    def __call__(self, doc):
        nlp = self.nlp

        #check if gold annotation file path has been set.
        if not hasattr(doc._, 'gold_annotation_file'):
            raise ValueError("No extension doc._.gold_annotation_file is present.")

        gold_annotations = ann_to_json(open(doc._.gold_annotation_file, 'r').read())

        # for label in set([label for _,_,label in [gold['entities'][key] for key in gold['entities']]]):
        Token.set_extension('gold_label', default="O", force=True)
        # for token in doc:
        #     print(token.text, token.idx)
        for e_start, e_end, e_label in [gold_annotations['entities'][key] for key in gold_annotations['entities']]:

            span = doc.char_span(e_start, e_end)
            if span is None:
                self.none_count += 1
                print(self.none_count)
            fixed_span = self.find_span(e_start, e_end, e_label, span, doc)
            if fixed_span is not None:
                if span is None:
                    print(fixed_span.text)
                for token in fixed_span:
                    if e_label in self.labels or not self.labels:
                        token._.set('gold_label', e_label)

        return doc




