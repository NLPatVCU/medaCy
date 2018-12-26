
class DataFile:
    """DataFile wraps all relevent information needed to manage a text document and it's corresponding annotation"""

    def __init__(self, file_name, raw_text_file_path, annotation_file_path, metamapped_path=None):
        """
        Wrapps a file and it's corresponding annotation in a single DataFile object
        :param file_name: the name of the file
        :param raw_text_file_path: the file path in memory of the raw text of the file
        :param annotation_file_path: the file path in memory of the annotation of the file
        """
        self.file_name = file_name
        self.raw_path = raw_text_file_path
        self.ann_path = annotation_file_path
        self.metamapped_path = metamapped_path

    def __repr__(self):
        return self.file_name

    def __str__(self):
        return self.file_name