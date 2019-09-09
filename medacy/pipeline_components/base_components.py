from abc import ABC, abstractmethod


class BaseOverlayer(ABC):
    """
    A base medaCy feature overlayer, which adds new annotations to spaCy Token objects.
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

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.component_name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Adds the new annotation(s) to spaCy Token objects in a given Doc object."""
        pass


class BaseLearner(ABC):

    @abstractmethod
    def fit(self, x_data, y_data):
        pass


class BaseTokenizer(ABC):
    """Determines where word boundaries should be for the tokenization process."""

    def add_exceptions(self, exceptions):
        """
        Adds exception for tokenizer to ignore.
        :param exceptions: an array of terms to not split on during tokenizers
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _get_prefix_regex(self):
        """Custom prefix tokenizers rules"""
        pass

    @abstractmethod
    def _get_infix_regex(self):
        """Custom infix tokenizers rules"""
        pass

    @abstractmethod
    def _get_suffix_regex(self):
        """Custom suffix tokenizers rules"""
        pass
