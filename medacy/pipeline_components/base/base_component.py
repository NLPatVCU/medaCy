from abc import ABC, abstractmethod


class BaseComponent(ABC):
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
            assert isinstance(component, BaseComponent), "Dependencies must be other components."

        self.dependencies = dependencies

    @abstractmethod
    def __call__(self, doc):
        """
        Overlays features onto a Doc object and/or the Token objects it contains
        :param doc: a spaCy Doc object (not a string of the doc text)
        :return: the Doc object that was passed to it (which was modified by this call)
        """
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.component_name
