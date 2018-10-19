
from ..metamap.metamap import MetaMap
from spacy.tokens import Token

class MetaMapAnnotatorComponent():
    """
    A pipeline component for SpaCy that overlays metamap output as token attributes
    """
    name = "metamap_annotator"

    def __init__(self, nlp):
        self.nlp = nlp
        self.metamap = MetaMap()


    def __call__(self, doc):
        metamap = self.metamap
        nlp = self.nlp

        #check if pre-metamapped file is associated with the doc
        #TODO Should this raise an exception instead? Metamapping should be preprocessing possibly
        if hasattr(doc._, 'metamapped_file'):
            metamap_dict = metamap.load(doc._.metamapped_file)
        else:
            metamap_dict = metamap.map_text(doc.text)


        mapped_terms = metamap.extract_mapped_terms(metamap_dict) #parse terms out of mappings dictionary



        semantic_type_labels = ['orch', 'phsu', 'inch', 'bacs', 'patf' ]
        semantic_type_labels += ['aapp', 'antb', 'sosy', 'dsyn', 'fndg','qlco', 'patf']


        spans = [] #for displaying NER output with displacy


        #Overlays if the given semantic type is set.
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




