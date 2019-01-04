import logging
from spacy.tokens import Token, Span
from spacy.matcher import PhraseMatcher
from ..base import BaseComponent

class LexiconComponent(BaseComponent):


    name = "lexicon_component"
    dependencies = []


    def __init__(self, spacy_pipeline, lexicon):
        """
        Initializes pipeline component that takes in an array of terms in a lexicon and a label to apply
        to matching terms in the document.
        :param spacy_pipeline: an instance of a spacy language pipeline
        :param lexicon: Dictionary containing labels and the corresponding array of terms to match to that label
        """
        super().__init__(self.name, self.dependencies)
        assert lexicon != {}, "The provided lexicon can not be empty"
        self.nlp = spacy_pipeline
        self.lexicon = lexicon

    def __call__(self, doc):
        """

        :param doc:
        :return:
        """
        logging.debug("Called Lexicon Component")

        matcher = PhraseMatcher(self.nlp.vocab, max_length=6)
        for label in self.lexicon:
            Token.set_extension('feature_is_' + label, default=False, force=True)
            patterns = [self.nlp.make_doc(term) for term in self.lexicon[label]]
            logging.debug(patterns)
            matcher.add(label, None, *patterns)
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = Span(doc, start, end)
            logging.debug(span)
            if span is not None:
                logging.debug('Lexicon term matched: %s Label: %s' % (span.text, self.nlp.vocab.strings[match_id]))
                for token in span:
                    token._.set('feature_is_' + self.nlp.vocab.strings[match_id], True)

        return doc