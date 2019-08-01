from abc import ABC, abstractmethod
from medacy.pipeline_components.base import BaseComponent

class BasePipeline(ABC):
    """
    An abstract wrapper for a Medical NER Pipeline
    """

    def __init__(self, pipeline_name, spacy_pipeline=None, description=None, creators="", organization=""):
        """
        Initializes a pipeline
        :param pipeline_name: The name of the pipeline
        :param spacy_pipeline: the corresponding spacy pipeline (language) to utilize.
        :param description: a description of the pipeline
        :param creator: the creator of the pipeline
        :param organization: the organization the pipeline creator belongs to
        """
        self.pipeline_name = pipeline_name
        self.spacy_pipeline = spacy_pipeline
        self.description = description
        self.creators = creators
        self.organization = organization

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
        pass

    def get_language_pipeline(self):
        """
        Retrieves the associated spaCy Language pipeline that the medaCy pipeline wraps.
        :return: spacy_pipeline
        """
        return self.spacy_pipeline

    def add_component(self, component, *argv, **kwargs):
        """
        Adds a given component to pipeline
        :param component: a subclass of BaseComponent
        """

        current_components = [component_name for component_name, proc in self.spacy_pipeline.pipeline]
        #print("Current Components:", current_components)
        dependencies = [x for x in component.dependencies]
        #print("Dependencies:",dependencies)

        # No need to change anything if the component is already there
        if component.name in current_components:
            # "%s is already in the pipeline." % component.name
            return

        for dependent in dependencies:
            assert dependent in current_components, "%s depends on %s but it hasn't been added to the pipeline" % (component, dependent)

        self.spacy_pipeline.add_pipe(component(self.spacy_pipeline, *argv, **kwargs))

    def get_components(self):
        """
        Retrieves a listing of all components currently in the pipeline.
        :return: a list of components inside the pipeline.
        """
        return [component_name for component_name, _ in self.spacy_pipeline.pipeline if component_name != 'ner']

    def __call__(self, doc, predict=False):
        """
        Passes a single document through the pipeline.
        All relevant document attributes should be set prior to this call.
        :param doc: the document to annotate over
        :return: the annotated document
        """

        for component_name, proc in self.spacy_pipeline.pipeline:
            if predict and component_name == "gold_annotator":
                continue
            doc = proc(doc)
            if component_name == 'ner':
                # remove labeled default entities
                doc.ents = []

        return doc

    def get_pipeline_information(self):
        """
        Retrieves information about the current pipeline in a structured dictionary
        :return: a json dictionary containing information
        """
        information = {
            'components': [component_name for component_name, _ in self.spacy_pipeline.pipeline
                           if component_name != 'ner'], #ner is the default ner component of spacy that is not utilized.
            'learner_name': self.get_learner()[0],
            'description': self.description,
            'pipeline_name': self.pipeline_name,
            'pipeline_creators': self.creators,
            'pipeline_creator_organization': self.organization
        }

        return information
