import logging
import spacy
import scispacy

from spacy.tokens import Token
from scispacy.umls_linking import UmlsEntityLinker
from medacy.pipeline_components.base.base_component import BaseComponent

linker = UmlsEntityLinker(resolve_abbreviations=True)

class AllenCUIComponent(BaseComponent):

    name = 'allen_cui_component'
    dependencies = []
    
    def __init__(self, spacy_pipeline):
        super().__init__(self.name, self.dependencies)
        self.nlp = spacy_pipeline
        self.sci = spacy.load("en_core_sci_sm")
        self.sci.tokenizer = self.nlp.tokenizer
        self.sci.add_pipe(linker)

    def __call__(self, doc):
        logging.debug("Called Allen CUI Component")
        Token.set_extension("allen_cui_component", default="DEFAULT", force=True)

        sci_doc = self.sci(doc)
        doc_pairs = zip(doc, sci_doc)
        for token, entity in doc_pairs:
            try:
                token._.set("allen_cui_component", entity._.umls_ents[0][0][0])
            except:
                pass

        return doc
