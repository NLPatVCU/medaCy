from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """
    A base medacy pipeline component that wraps over a spacy component
    """

    def __init__(self, component_name, dependencies=[]):
        self.component_name = component_name

        for component in dependencies:
            assert isinstance(component, BaseComponent), "Dependencies must be other components."

        self.dependencies = dependencies

    def get_component_name(self):
        return self.component_name

    def get_component_dependencies(self):
        return self.dependencies

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.get_component_name()
