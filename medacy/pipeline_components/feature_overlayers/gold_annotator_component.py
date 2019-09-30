import logging

from spacy.tokens import Token

from medacy.data.annotations import Annotations
from medacy.pipeline_components.base.base_component import BaseComponent


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
        Token.set_extension('gold_label', default="O", force=True)

    def find_span(self, start, end, label, span, doc):
        """
        Greedily searches characters around word to find a valid set of tokens the annotation likely corresponds to.
        :param start: index of token start
        :param end: index of token end
        :param label:
        :param span:
        :param doc:
        :return:
        """
        greedy_searched_span = doc.char_span(start, end)
        if greedy_searched_span is not None:
            return greedy_searched_span

        greedy_searched_span = doc.char_span(start, end-1) #annotation may have extended over an ending blank space
        if greedy_searched_span is not None:
            return greedy_searched_span

        #No clue - increase boundaries incrementally until a valid span is found.
        i = 0
        while greedy_searched_span is None and i <= 20:
            if i % 2 == 0:
                end += 1
            else:
                start-=1
            i+=1
            greedy_searched_span = doc.char_span(start, end)

        return greedy_searched_span



    def __call__(self, doc):
        nlp = self.nlp

        if hasattr(doc._, 'file_name'):
            logging.debug("%s: Called GoldAnnotator Component", doc._.file_name)

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #print document tokenization
            for token in doc:
                logging.debug(str(token))

        #check if gold annotation file path has been set.
        if not hasattr(doc._, 'gold_annotation_file'):
            raise ValueError("No extension doc._.gold_annotation_file is present.")

        gold_annotations = Annotations(doc._.gold_annotation_file)

        # for label in set([label for _,_,label in [gold['entities'][key] for key in gold['entities']]]):

        # for token in doc:
        #     print(token.text, token.idx)
        for e_label, e_start, e_end, _ in gold_annotations.get_entity_annotations():
            #print(e_label, e_start, e_end)
            if e_start > e_end:
                logging.critical("%s: Broken annotation - start is greater than end: (%i,%i,%s)",doc._.file_name, e_start, e_end, e_label)
                continue
            span = doc.char_span(e_start, e_end)

            if span is None:
                self.failed_overlay_count += 1
                self.failed_identifying_span_count += 1
                logging.warning("%s: Number of failed annotation overlays with current tokenizer: %i (%i,%i,%s)", doc._.file_name, self.failed_overlay_count, e_start, e_end, e_label)
            fixed_span = self.find_span(e_start, e_end, e_label, span, doc)
            if fixed_span is not None:
                if span is None:
                    logging.warning("%s: Fixed span (%i,%i,%s) into: %s", doc._.file_name, e_start, e_end, e_label,fixed_span.text)
                    self.failed_identifying_span_count -= 1
                for token in fixed_span:
                    if e_label in self.labels or not self.labels:
                        token._.set('gold_label', e_label)

            else: #annotation was not able to be fixed, it will be ignored - this is bad in evaluation.
                logging.warning("%s: Could not fix annotation: (%i,%i,%s)",doc._.file_name, e_start, e_end, e_label)
                logging.warning("%s: Total Failed Annotations: %i", doc._.file_name, self.failed_identifying_span_count)

        if self.failed_overlay_count > .3*len(gold_annotations.get_entity_annotations()) :
            logging.warning("%s: Annotations may mis-aligned as more than 30 percent failed to overlay: %s", doc._.file_name, doc._.gold_annotation_file)


        return doc




