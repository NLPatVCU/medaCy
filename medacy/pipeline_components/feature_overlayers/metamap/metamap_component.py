import logging
import warnings
import os

from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.base import BaseOverlayer
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap


def _get_metamapped_path(txt_file_path):
    """
    MedaCy assumes that for a given txt file, the related metamapped file exists with the same file name,
    the extension '.metamapped', and that the file is located in a subdirectory named 'metamapped';
    this function takes the full path to a txt file and returns the expected path of the metamapped file.
    :param txt_file_path: the full path to a txt file
    :return: the full path to the metamapped file
    """
    training_dir = os.path.dirname(txt_file_path)
    txt_file_name_ext = os.path.basename(txt_file_path)
    txt_file_name, _ = os.path.splitext(txt_file_name_ext)
    return os.path.join(training_dir, 'metamapped', txt_file_name + '.metamapped')


class MetaMapOverlayer(BaseOverlayer):
    """
    A pipeline component for spaCy that overlays Metamap output as token attributes
    """
    
    name = "metamap_annotator"
    dependencies = []

    def __init__(self, spacy_pipeline, metamap: MetaMap, cuis=True, semantic_type_labels=None, merge_tokens=False):
        """
        Initializes a pipeline component that annotates MetaMap output onto a spacy doc object.
        :param spacy_pipeline: an instance of a spacy language pipeline.
        :param metamap: an instance of MetaMap.
        :param cuis: Whether or not to overlay CUIs from MetaMap output.
        :param semantic_type_labels: list of semantic type labels to check for- generates a feature for each semantic type label.
        """
        super().__init__(self.name, self.dependencies)
        self.nlp = spacy_pipeline
        self.metamap = metamap
        self.cuis = cuis
        self.semantic_type_labels = semantic_type_labels or []
        self.merge_tokens = merge_tokens

    def __call__(self, doc):
        """
        Runs a document to the metamap_annotator pipeline component. This overlays rich medical features by utilizing
        MetaMap output and aligning it with a passed spacy Doc object. By medaCy conventions, each overlayed feature
        is available as a token extension starting with 'feature_'. This component overlays 'feature_cui' and a
        separate boolean feature for each semantic type to detect available under 'feature_is_{type}". This component
        was originally designed to increase recall on Drug entities hence by default 'feature_is_orch' and
        'feature_is_phsu' where orch and phsu are semantic types corresponding to organic chemicals and pharmalogical
        substances respectively.
        :param doc: spaCy Doc object to run through pipeline
        :return: the same Doc object
        """
        logging.debug("Called MetaMap Component")

        # register all extensions
        if self.cuis:
            Token.set_extension('feature_cui', default="-1", force=True) #cui feature
        for label in self.semantic_type_labels:  # is_semantic type features
            Token.set_extension('feature_is_' + label, default=False, force=True)

        if not hasattr(doc._, 'file_name'):
            metamap_dict = self.metamap.map_text(str(doc))
        elif doc._.file_name is None or doc._.file_name == 'STRING_INPUT':
            metamap_dict = self.metamap.map_text(str(doc))
        elif os.path.isfile(doc._.file_name):
            # Check if pre-metamapped file exists at expected location
            txt_file_path = doc._.file_name
            metamapped_path = _get_metamapped_path(txt_file_path)
            if not os.path.isfile(metamapped_path):
                warnings.warn(f"No metamapped file was found for '{txt_file_path}'; attempting to run MetaMap over document (results in slower runtime); ensure MetaMap is running")
                metamap_dict = self.metamap.map_text(str(doc))
            else:
                # This branch of the decision tree is reached if the file is already metamapped
                metamap_dict = self.metamap.load(metamapped_path)

        # TODO refactor second part of if statement when implementing live model prediction
        if metamap_dict == '' or metamap_dict['metamap'] is None:
            if hasattr(doc._, 'file_name'):
                warnings.warn(f"MetaMap produced no output for given file: {doc._.file_name}")
            warnings.warn("MetaMap failed")
            return doc

        mapped_terms = self.metamap.extract_mapped_terms(metamap_dict)  # parse terms out of mappings dictionary

        spans = []  # for displaying NER output with displacy

        # Overlays semantic type presence if the given semantic type is set in metamap span.
        for semantic_type_label in self.semantic_type_labels:

            entity_name = semantic_type_label
            self.nlp.entity.add_label(entity_name)  # register entity label

            entity_tags = self.metamap.get_term_by_semantic_type(mapped_terms, include=[semantic_type_label])
            entity_annotations = self.metamap.mapped_terms_to_spacy_ann(entity_tags, semantic_type_label)

            with doc.retokenize() as retokenizer:
                for start, end, label in entity_annotations:
                    span = doc.char_span(start, end, label=self.nlp.vocab.strings[entity_name])

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

        # Overlays CUI of each term
        if Token.has_extension('feature_cui'):
            with doc.retokenize() as retokenizer:
                for term in mapped_terms:
                    cui = term['CandidateCUI']
                    start, end = self.metamap.get_span_by_term(term)[0]
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

    def get_report(self):
        report = super().get_report() + '\n'
        report += f"\tcuis = {self.cuis}\n\tsemantic_type_labels = {self.semantic_type_labels}\n\tmerge_tokens = {self.merge_tokens}"
        return report
