import inspect
from abc import ABC, abstractmethod


class BaseOverlayer(ABC):
    """
    A base medacy pipeline component that wraps over a spaCy component
    """

    def __init__(self, component_name, dependencies):
        """
        :param component_name: The name of the component
        :param dependencies: Other components that this component depends on
        """
        self.component_name = component_name

        for component in dependencies:
            assert isinstance(component, BaseOverlayer), "Dependencies must be other components."

        self.dependencies = dependencies

    @abstractmethod
    def __call__(self, doc):
        """
        Overlays features onto a Doc object and/or the Token objects it contains
        :param doc: a spaCy Doc object (not a string of the doc text)
        :return: the Doc object that was passed to it (which was modified by this call)
        """
        pass

    def get_report(self):
        """
        Creates a report about the configuration of the overlayer instance; implementations in subclasses
        should contain all the information needed to reconstruct the instance
        :return: str
        """
        return f"{type(self)} at {inspect.getfile(type(self))}"

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.component_name
