from medacy.pipeline_components.metamap.metamap import MetaMap
from spacy.tokens import Token
import warnings,logging


class MetaMapComponent():
    """
    A pipeline component for SpaCy that overlays Metamap output as token attributes
    """
    name = "metamap_annotator"
    dependencies = []
    def __init__(self, nlp, metamap):
        self.nlp = nlp
        assert isinstance(metamap, MetaMap), "MetamapComponent requires a MetaMap instance as an argument."
        self.metamap = metamap


    def __call__(self, doc):
        logging.debug("Called MetaMap Component")
        metamap = self.metamap
        nlp = self.nlp

        #check if pre-metamapped file is associated with the doc
        if hasattr(doc._, 'metamapped_file'):
            metamap_dict = metamap.load(doc._.metamapped_file)
        else:
            logging.debug("Could not find metamap file for document.")
            metamap_dict = metamap.map_text(doc.text)

        if metamap_dict['metamap'] is None:
            if hasattr(doc._, 'metamapped_file'):
                warnings.warn("This metamap file is invalid and cannot be parsed in MetaMapComponent: %s \n Ignore this warning if this is a unittest - all may be fine." % doc._.metamapped_file)
            else:
                warnings.warn("Metamapping text on the fly failed - aborting. Try to pre-metamap with DataLoader.")
            return doc

        mapped_terms = metamap.extract_mapped_terms(metamap_dict) #parse terms out of mappings dictionary



        semantic_type_labels = ['orch', 'phsu']
        # semantic_type_labels += ['inch', 'bacs', 'patf' ,'aapp', 'antb', 'sosy', 'dsyn', 'fndg','qlco', 'patf']


        spans = [] #for displaying NER output with displacy


        #Overlays semantic type presence if the given semantic type is set in metamap span.
        for semantic_type_label in semantic_type_labels:
            Token.set_extension('feature_is_' + semantic_type_label, default=False, force=True)  # register extension to token

            entity_name = 'Drug'
            nlp.entity.add_label(entity_name) #register entity label

            entity_tags = metamap.get_term_by_semantic_type(mapped_terms, include=[semantic_type_label])
            entity_annotations = metamap.mapped_terms_to_spacy_ann(entity_tags, semantic_type_label)


            for start, end, label in [entity_annotations['entities'][key] for key in entity_annotations['entities'].keys()]:
                span = doc.char_span(start, end, label=nlp.vocab.strings[entity_name])

                #TODO spans are none when indices and token boundaries don't line up. This shouldn't happen here
                #TODO but needs to be investigated.

                if span is not None:
                    spans.append(span)
                    for token in span:
                        token._.set('feature_is_' + label, True)

        #adds labels for displaying NER output with displacy.
        for span in spans:
            doc.ents = list(doc.ents) + [span]

        #Overlays CUI of each term
        Token.set_extension('feature_cui', default="-1", force=True)
        for term in mapped_terms:
            cui = term['CandidateCUI']
            start, end = metamap.get_span_by_term(term)[0]
            span = doc.char_span(start, end)
            if span is not None:
                for token in span:
                    token._.set('feature_cui', cui)



        return doc




