import os
from medacy.tools import DataFile
from pkg_resources import resource_listdir, resource_string


def load_END():
    """
    Loads the Engineered Nanomedicine Database (END) for testing purposes. This dataset is a collection of annotated
    FDA nanomedicine drug labels containing 28 unique entities.
    :return: a tuple containing an generator of (raw_text, ann_file) tuples and an array of target entities
    """
    package_name = 'medacy.tests.data'
    resource_directory = 'end'
    contents = resource_listdir(package_name, resource_directory)
    entities = resource_string('medacy.tests.data', resource_directory+'/END.types').decode('utf-8').split("\n")
    files = [file[:-4] for file in contents if file.endswith('.ann')]
    files = ((file_name,
              resource_string('medacy.tests.data', resource_directory+"/%s.txt" % file_name).decode('utf-8'),
              resource_string('medacy.tests.data', resource_directory + "/%s.ann" % file_name).decode('utf-8')
              )
             for file_name in files)
    return files, entities