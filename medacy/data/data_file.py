import os


class DataFile:
    """
    DataFile wraps all relevant information needed to manage a text document and it's corresponding annotation. Specifically,
    a Datafile keeps track of the filepath of the raw text, annotation file, and metamapped file for each document.

    :ivar file_name: the name of the file being represented
    :ivar txt_path: the text document corresponding to this file
    :ivar ann_path: the annotations of the text document
    :ivar metamapped_path: the metamap file
    """

    def __init__(self, file_name, raw_text_file_path, annotation_file_path, metamapped_path=None):
        """
        Wraps a file and it's corresponding annotation in a single DataFile object
        :param file_name: the name of the file
        :param raw_text_file_path: the file path in memory of the raw text of the file
        :param annotation_file_path: the file path in memory of the annotation of the file
        """
        self._file_name = file_name
        self._raw_path = raw_text_file_path
        self._ann_path = annotation_file_path
        self._metamapped_path = metamapped_path

    @property
    def file_name(self):
        return self._file_name

    @property
    def txt_path(self):
        return self._raw_path

    @property
    def ann_path(self):
        return self._ann_path

    @property
    def metamapped_path(self):
        return self._metamapped_path

    @metamapped_path.setter
    def metamapped_path(self, path):
        if not os.path.isfile(path): raise FileNotFoundError(f"'{path}'' is not a file")
        self._metamapped_path = path

    def __repr__(self):
        return f"{type(self).__name__}({self._file_name}, {self._raw_path}, {self._ann_path}, {self._metamapped_path})"

    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return self._file_name == other._file_name if isinstance(other, DataFile) else False

    def __hash__(self):
        return hash(self._file_name)
