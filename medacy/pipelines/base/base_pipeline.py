import inspect
import time
from abc import ABC, abstractmethod

from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorOverlayer


class BasePipeline(ABC):
    """
    An abstract wrapper for a Medical NER Pipeline
    """

    def __init__(self, entities, spacy_pipeline, cuda_device=-1):
        """
        Initializes a pipeline
        :param entities: a list of entities, or an empty list if the pipeline is for a model that has already been fitted
        :param spacy_pipeline: the corresponding spacy pipeline (language) to utilize.
        :param cuda_device: the GPU to use, if any (defaults to -1 for using the CPU)
        """
        self.entities = entities
        self.spacy_pipeline = spacy_pipeline
        self.cuda_device = cuda_device
        self.overlayers = []  # Stores feature overlayers

        self.spacy_pipeline.tokenizer = self.get_tokenizer()  # set tokenizer

        if entities:
            self.add_component(GoldAnnotatorOverlayer, entities)

        # The following code was causing GPU errors because you cannot specify which GPU spaCy will use;
        # You may uncomment this code if you know you have access to the GPU that spaCy will use.

        # if cuda_device >= 0:
        #     spacy.require_gpu()

    @abstractmethod
    def get_tokenizer(self):
        """Returns an instance of a tokenizer"""
        pass

    @abstractmethod
    def get_learner(self):
        """Retrieves an instance of a sci-kit learn compatible learning algorithm."""
        pass

    @abstractmethod
    def get_feature_extractor(self):
        """Returns an instant of FeatureExtractor with all configs set."""
        pass

    def add_component(self, component, *argv, **kwargs):
        """
        Adds a given component to pipeline
        :param component: a subclass of BaseOverlayer
        """

        current_components = [component_name for component_name, proc in self.spacy_pipeline.pipeline]

        assert component.name not in current_components, "%s is already in the pipeline." % component.name

        for dependent in component.dependencies:
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
        :param self:
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

    def get_report(self):
        """
        Generates a report about the pipeline class's configuration
        :return: str
        """

        # Get data about these components
        learner_name, learner = self.get_learner()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        # Start the report with the name of the class and the docstring
        report = f"{type(self).__name__}\n{self.__doc__}\n\n"

        report += f"Report created at {time.asctime()}\n\n"
        report += f"Entities: {self.entities}\n\n"

        # Print data about the feature overlayers
        if self.overlayers:
            report += "Feature Overlayers:\n\n"
            report += "\n\n".join(o.generate_report() for o in self.overlayers) + '\n\n'

        # Print data about the feature extractor
        report += f"Feature Extractor: {type(feature_extractor).__name__} at {inspect.getfile(type(feature_extractor))}\n"
        report += f"\tWindow Size: {feature_extractor.window_size}\n"
        report += f"\tSpaCy Features: {feature_extractor.spacy_features}\n"


        # Print the name and location of the remaining components
        report += f"Learner: {learner_name} at {inspect.getfile(type(learner))}\n"
        report += f"Tokenizer: {type(tokenizer).__name__} at {inspect.getfile(type(tokenizer))}\n"

        return report
