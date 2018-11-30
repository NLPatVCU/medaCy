from abc import ABC, abstractmethod
from ...pipeline_components.base import BaseComponent

class BasePipeline(ABC):
    """
    An abstract wrapper for a Medical NER Pipeline
    """

    def __init__(self,pipeline_name, spacy_pipeline=None):
        self.pipeline_name = pipeline_name
        self.spacy_pipeline = spacy_pipeline


    @abstractmethod
    def get_tokenizer(self):
        """
        Returns an instance of a tokenizer
        :return:
        """
        pass

    @abstractmethod
    def get_learner(self):
        """
        Retrieves an instance of a sci-kit learn compatible learning algorithm.
        :return: model
        """
        pass

    @abstractmethod
    def get_feature_extractor(self):
        """
        Returns an instant of FeatureExtractor with all configs set.
        :return: An instant of FeatureExtractor
        """


    def get_language_pipeline(self):
        """
        Retrieves the associated spaCy Language pipeline that the medaCy pipeline wraps.
        :return: spacy_pipeline
        """
        return self.spacy_pipeline

    def add_component(self, component, *argv):
        """
        Adds a given component to pipeline
        :param component: a subclass of BaseComponent
        """

        current_components = [component_name for component_name, proc in self.spacy_pipeline.pipeline]
        #print("Current Components:", current_components)
        dependencies = [x for x in component.dependencies]
        #print("Dependencies:",dependencies)

        assert component.name not in current_components, "%s is already in the pipeline." % component.name

        for dependent in dependencies:
            assert dependent in current_components, "%s depends on %s but it hasn't been added to the pipeline" % (component, dependent)



        self.spacy_pipeline.add_pipe(component(self.spacy_pipeline, *argv))

    def __call__(self, doc, predict=False):
        """
        Passes a single document through the pipeline.
        All relevant document attributes should be set prior to this call.
        :param self:
        :param doc:
        :return:
        """

        for component_name, proc in self.spacy_pipeline.pipeline:
            if predict and component_name == "gold_annotator":
                continue
            doc = proc(doc)
            if component_name == 'ner':
                # remove labeled default entities
                doc.ents = []

        return doc






