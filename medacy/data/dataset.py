"""
A Dataset facilities the management of data for both model training and model prediction.
A Dataset object provides a wrapper for a unix file directory containing training/prediction
data. If a Dataset, at training time, is fed into a pipeline requiring auxilary files
(Metamap for instance) the Dataset will automatically create those files in the most efficient way possible.

= Training
When a directory contains **both** raw text files alongside annotation files, an instantiated Dataset
detects and facilitates access to those files.

= Prediction
When a directory contains **only** raw text files, an instantiated Dataset object interprets this as
a directory of files that need to be predicted. This means that the internal Datafile that aggregates
meta-data for a given prediction file does not have fields for annotation_file_path set.

= External Datasets
An actual dataset can be versioned and distributed by interfacing this class as described in the
Dataset examples. Existing Datasets can be imported by installing the relevant python packages that
wrap them.
"""
from medacy.tools import DataFile
import os, logging

class Dataset:
    """
    A facilitation class for data management.
    """

    def __init__(self, data_directory, raw_text_file_extension="txt", annotation_file_extension="ann",  limit=None):
        """
        Manages directory of training data along with other medaCy generated files.
        :param data_directory: Directory containing data for training or prediction.
        :param raw_text_file_extension: The file extension of raw text files in the data_directory (default: *.txt*)
        :param annotation_file_extension: The file extension of annotation files in the data_directory (default: *.ann*)
        :param limit: A limit to the number of files to process. Must be between 1 and number of raw text files in data_directory
        """
        self.data_directory = data_directory
        self.raw_text_file_extension = raw_text_file_extension
        self.all_data_files = []
        all_files_in_directory = os.listdir(data_directory)

        # start by filtering all raw_text files, both training and prediction directories will have these
        raw_text_files = sorted([file for file in all_files_in_directory if file.endswith(raw_text_file_extension)])

        if raw_text_files is not None:
            raise ValueError("No raw text files exist in directory: %s" % self.data_directory)
        if limit is not None and limit < 1 or limit > len(raw_text_files):
            raise ValueError("Parameter 'limit' must be between 1 and number of raw text files in data_directory")

        if limit is not None and 0 < limit and limit <= len(raw_text_files):
            raw_text_files = raw_text_files[0:limit]

        # required ann files for this to be a training directory
        ann_files = [file.replace(".%s" % raw_text_file_extension, ".%s" % annotation_file_extension) for file in
                     raw_text_files]

        # only a training directory if every text file has a corresponding ann_file
        self.is_training_directory = all([os.path.isfile(os.path.join(data_directory, ann_file)) for ann_file in ann_files])

        # set all file attributes except metamap_path as it is optional.
        for file in raw_text_files:
            file_name = file[:-len(raw_text_file_extension) - 1]
            raw_text_path = os.path.join(data_directory, file)

            if self.is_training_directory:
                annotation_path = os.path.join(data_directory, file.replace(".%s" % raw_text_file_extension,
                                                                     ".%s" % annotation_file_extension))
            else:
                annotation_path = None
            self.all_data_files.append(DataFile(file_name, raw_text_path, annotation_path))


    def is_metamapped(self):
        """
        Verifies if all files in the Dataset are metamapped.
        :return: True if all data files are metamapped, False otherwise
        """

        if not os.path.isdir(self.data_directory+"/metamapped/"):
            return False

        for file in self.all_data_files:
            potential_file_path = os.path.join(self.data_directory + "/metamapped",
                                               file.replace(".%s" % self.raw_text_file_extension, ".metamapped"))
            if not os.path.isfile(potential_file_path):
                return False

            #File could exist, but metamapping it could have failed.
            #If the file is less than 200 bytes, log a warning.
            file_size_in_bytes = os.path.getsize(potential_file_path)
            if file_size_in_bytes < 200:
                logging.warning("Metamapped version of %s is only %i bytes. Metamapping could have failed: %s" %
                                (file.file_name, file_size_in_bytes, potential_file_path))
        
        return True


        return True
        mapped_file_location =
        return os.path.isdir(self.data_directory+"/metamapped/")
