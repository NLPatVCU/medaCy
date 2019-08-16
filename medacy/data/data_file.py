class DataFile:
    """
    DataFile wraps all relevant information needed to manage a text document and it's corresponding annotation. Specifically,
    a Datafile keeps track of the filepath of the raw text, annotation file, and metamapped file for each document.
    """

    def __init__(self, file_name, raw_text_file_path, annotation_file_path, metamapped_path=None):
        """
        Wraps a file and it's corresponding annotation in a single DataFile object

        :param file_name: the name of the file
        :param raw_text_file_path: the file path in memory of the raw text of the file
        :param annotation_file_path: the file path in memory of the annotation of the file
        """
        self.file_name = file_name
        self.raw_path = raw_text_file_path
        self.ann_path = annotation_file_path
        self.metamapped_path = metamapped_path

    def get_text_path(self):
        """
        Retrieves the file path of the text document that can be read from.

        :return: file path of the text document.
        """
        return self.raw_path

    def get_annotation_path(self):
        """
        Retrieves the file path of the annotation document that can be read from.

        :return: file path of the annotation document.
        """
        return self.ann_path

    def get_metamapped_path(self):
        """
        Retrieves the file path of the metamap output document that can be read from. This is only set if the document
        is metamapped, other it is None.

        :return: file path of the metamap output document.
        """
        return self.metamapped_path

    def __repr__(self):
        return self.file_name

    def __str__(self):
        return self.file_name