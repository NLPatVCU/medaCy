from abc import ABC


class BaseComponent(ABC):
    """
    A base medacy pipeline component that wraps over a spacy component
    """

    def __init__(self, component_name="DEFAULT_COMPONENT_NAME", dependencies=[]):
        """

        :param component_name: The name of the component
        :param dependencies: Other components that this component depends on
        """
        self.component_name = component_name

        for component in dependencies:
            assert isinstance(component, BaseComponent), "Dependencies must be other components."

        self.dependencies = dependencies

    def get_component_name(self):
        return self.component_name

    def get_component_dependencies(self):
        """
        Retrieves a list of dependencies this component has.
        :return: a list of component dependencies
        """
        return self.dependencies

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.get_component_name()
