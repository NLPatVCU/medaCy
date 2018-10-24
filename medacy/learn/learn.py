"""
Builds a model with a given medacy pipeline and dataset
"""
from ..pipelines.base import BasePipeline
from ..tools import DataLoader
from ..pipeline_components import MetaMap
from ..learn import FeatureExtractor

class Learner:
    def __init__(self, medacy_pipeline, data_loader, metamap=None):
        """

        :param medacy_pipeline: A sub-class of BasePipeline such as ClinicalPipeline
        :param data_loader: An instance of DataLoader
        :param metamap: an instance of metamap, that if present will cause the data_loader to directory.
        """


        assert isinstance(medacy_pipeline, BasePipeline), "Pipeline must be a medaCy pipeline that interfaces medacy.pipelines.base.BasePipeline"
        assert isinstance(data_loader, DataLoader), "Must give an instance of DataLoader into the Learner"

        self.medacy_pipeline = medacy_pipeline
        self.data_loader = data_loader

        if metamap is not None and isinstance(metamap, MetaMap):
            data_loader.metamap(metamap)


        nlp = medacy_pipeline.spacy_pipeline
        feature_extractor = FeatureExtractor()
        X_data = []
        y_data = []

        for data_file in data_loader.get_files():

            with open(data_file.raw_path, 'r') as raw_text:
                doc = nlp.make_doc(raw_text.read())

            #Link ann_path to doc
            doc.set_extension('gold_annotation_file', default=data_file.ann_path, force=True)

            #Link metamapped file to doc for use in MetamapComponent if relevant
            if data_file.metamapped_path is not None:
                doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

            #run 'er through
            doc = medacy_pipeline(doc)

            """
            The document has now be run through the pipeline. All annotations are overlayed - pull features.
            """

            features, labels = feature_extractor(doc)


    def train(self):
        pass











