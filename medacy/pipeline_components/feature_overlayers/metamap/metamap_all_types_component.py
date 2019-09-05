import logging
import re
import warnings

from spacy.tokens import Token

from medacy.pipeline_components.metamap.metamap import MetaMap
from medacy.pipeline_components.base.base_component import BaseComponent


class MetaMapAllTypesComponent(BaseComponent):
    """A pipeline component that uses all semantic types found in a given document"""

    name = "metamap_all_types_component"
    dependencies = []

    def __init__(self, spacy_pipeline, metamap, cuis=True, merge_tokens=False):
        """
        Initializes a pipeline component that annotates MetaMap output onto a spaCy doc object.
        :param spacy_pipeline: an instance of a spaCy language pipeline.
        :param metamap: an instance of MetaMap.
        :param cuis: Overlay CUIS from metamap output - one feature taking on multiple categorical values representing cuis.
        """

        if not isinstance(metamap, MetaMap):
            raise TypeError("MetamapComponent requires a MetaMap instance as an argument, but is %s" % repr(metamap))

        super().__init__(self.name, self.dependencies)
        self.nlp = spacy_pipeline
        self.metamap = metamap
        self.cuis = cuis
        self.merge_tokens = merge_tokens

    @staticmethod
    def get_all_semantic_types(mm_file_path):
        """Given a metamap file path, returns a list of all semantic types that appear in that file."""
        types = set()

        with open(mm_file_path) as f:
            mm_text = f.read()

        for m in re.findall(r"'SemType': '[a-z]+'", mm_text):
            match = m.string[12, -1]
            types.update(match)

        print(types)
        return list(types)

    def __call__(self, doc):
        """
        Runs a document to the metamap_annotator pipeline component. This overlays rich medical features by utilizing
        MetaMap output and aligning it with a passed spaCy Doc object. By medaCy conventions, each overlayed feature
        is available as a token extension starting with 'feature_'. This component overlays 'feature_cui' and a
        separate boolean feature for each semantic type to detect available under 'feature_is_{type}".
        :param doc: document to run through pipeline
        :return:
        """
        logging.debug("Called MetaMap Component")
        metamap = self.metamap
        nlp = self.nlp

        #check if pre-metamapped file has been assigned to the document
        if hasattr(doc._, 'metamapped_file'):
            metamap_dict = metamap.load(doc._.metamapped_file)
        else:
            if hasattr(doc._, 'file_name'):
                logging.debug("%s: Could not find metamap file for document." % doc._.file_name)
            metamap_dict = metamap.map_text(doc.text) #TODO metamap.map_text is broken currently

        if not hasattr(doc._, 'file_name'): #TODO REMOVE when implemnting live model prediction
            return doc

        # TODO refactor second part of if statement when implementing live model prediction
        if metamap_dict == '' or metamap_dict['metamap'] is None:
            if hasattr(doc._, 'metamapped_file'):
                warnings.warn("%s: This metamap file is invalid and cannot be parsed in MetaMapComponent: %s \n Ignore this warning if this is a unittest - all may be fine." % (doc._.file_name,doc._.metamapped_file))
            else:
                warnings.warn("Metamapping text on the fly failed - aborting. Try to pre-metamap with DataLoader.")
            return doc

        mapped_terms = metamap.extract_mapped_terms(metamap_dict) #parse terms out of mappings dictionary

        spans = [] #for displaying NER output with displacy

        semantic_type_labels = self.get_all_semantic_types(doc._.metamapped_file)

        #register all extensions
        if self.cuis:
            Token.set_extension('feature_cui', default="-1", force=True) #cui feature
        for semantic_type_label in semantic_type_labels: #is_semantic type features
            Token.set_extension('feature_is_' + semantic_type_label, default=False, force=True)

        #Overlays semantic type presence if the given semantic type is set in metamap span.
        for semantic_type_label in semantic_type_labels:

            entity_name = semantic_type_label
            nlp.entity.add_label(entity_name) #register entity label

            entity_tags = metamap.get_term_by_semantic_type(mapped_terms, include=[semantic_type_label])
            entity_annotations = metamap.mapped_terms_to_spacy_ann(entity_tags, semantic_type_label)

            with doc.retokenize() as retokenizer:
                for start, end, label in entity_annotations['entities'].values():
                    span = doc.char_span(start, end, label=nlp.vocab.strings[entity_name])

                    #TODO spans are none when indices and token boundaries don't line up.
                    if span not in spans:
                        if span is not None:
                            logging.debug("Found from metamap: (label=%s,raw_text=\"%s\",location=(%i, %i))" % (label,span.text, start, end ) )
                            spans.append(span)
                            for token in span:
                                token._.set('feature_is_' + label, True)
                            if self.merge_tokens:
                                try:
                                    retokenizer.merge(span)
                                except BaseException:
                                    continue
                        else:
                            logging.debug("Metamap span could not be overlayed due to tokenization mis-match: (%i, %i)" % (start, end))

        #adds labels for displaying NER output with displacy.

        # for span in spans:
        #     try:
        #         doc.ents = list(doc.ents) + [span]
        #     except ValueError as error:
        #         logging.warning(str(error)) #This gets called when the same token may match multiple semantic types

        #Overlays CUI of each term
        if Token.has_extension('feature_cui'):
            with doc.retokenize() as retokenizer:
                for term in mapped_terms:
                    cui = term['CandidateCUI']
                    start, end = metamap.get_span_by_term(term)[0]
                    span = doc.char_span(start, end)
                    if span is not None:
                        for token in span:
                            token._.set('feature_cui', cui)
                        if self.merge_tokens:
                            try:
                                retokenizer.merge(span)
                            except BaseException:
                                continue

        return doc
