import logging
import re

from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer


class TableMatcherOverlayer(BaseOverlayer):

    name='table_matcher_component'
    dependencies=[]

    def __init__(self, spacey_pipeline):
        """

        :param spacey_pipeline:
        """
        super().__init__(self.name, self.dependencies)
        self.nlp = spacey_pipeline


    def __call__(self, doc):
        """
        Runs the document through the Table Matcher Component.  Uses regex patterns to identify terms that
        likely came from a table in the unstructured text.
        :param doc:
        :return:
        """
        logging.debug("Called Table Matcher Component")
        TABLE_PATTERN = re.compile(r'^(.*?)[ \t]{3,}\d+')
        Token.set_extension('feature_is_from_table', default=False, force=True)

        for match in re.finditer(TABLE_PATTERN, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span is None:
                continue
            for token in span:
                token._.set('feature_is_from_table', True)

        return doc
