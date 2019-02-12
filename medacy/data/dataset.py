"""
A Dataset facilities the management of data for both model training and model prediction.
A Dataset object provides a wrapper for a unix file directory containing training/prediction
data. If a Dataset, at training time, is fed into a pipeline requiring auxilary files
(Metamap for instance) the Dataset will automatically create those files in the most efficient way possible.

# Training
When a directory contains **both** raw text files alongside annotation files, an instantiated Dataset
detects and facilitates access to those files.

# Prediction
When a directory contains **only** raw text files, an instantiated Dataset object interprets this as
a directory of files that need to be predicted. This means that the internal Datafile that aggregates
meta-data for a given prediction file does not have fields for annotation_file_path set.

# External Datasets
An actual dataset can be versioned and distributed by interfacing this class as described in the
Dataset examples. Existing Datasets can be imported by installing the relevant python packages that
wrap them.
"""
from medacy.tools import DataFile
from joblib import Parallel, delayed
import os, logging, multiprocessing, math, json, importlib


class Dataset:
    """
    A facilitation class for data management.
    """

    def __init__(self, data_directory,
                 raw_text_file_extension="txt",
                 annotation_file_extension="ann",
                 metamapped_files_directory = None,
                 data_limit=None):
        """
        Manages directory of training data along with other medaCy generated files.
        :param data_directory: Directory containing data for training or prediction.
        :param raw_text_file_extension: The file extension of raw text files in the data_directory (default: *.txt*)
        :param annotation_file_extension: The file extension of annotation files in the data_directory (default: *.ann*)
        :param metamapped_files_directory: Location to store metamapped files (default: a sub-directory named *metamapped*)
        :param data_limit: A limit to the number of files to process. Must be between 1 and number of raw text files in data_directory
        """
        self.data_directory = data_directory
        self.raw_text_file_extension = raw_text_file_extension
        self.all_data_files = []

        if metamapped_files_directory is not None:
            self.metamapped_files_directory = metamapped_files_directory
        else:
            self.metamapped_files_directory = os.path.join(self.data_directory, 'metamapped')

        all_files_in_directory = os.listdir(self.data_directory)

        # start by filtering all raw_text files, both training and prediction directories will have these
        raw_text_files = sorted([file for file in all_files_in_directory if file.endswith(raw_text_file_extension)])

        if raw_text_files is None:
            raise ValueError("No raw text files exist in directory: %s" % self.data_directory)

        if data_limit is not None:
            self.data_limit = data_limit
        else:
            self.data_limit = len(raw_text_files)

        if self.data_limit < 1 or self.data_limit > len(raw_text_files):
            raise ValueError("Parameter 'data_limit' must be between 1 and number of raw text files in data_directory")

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

        #If directory is already metamapped, use it.
        if self.is_metamapped():
            for data_file in self.all_data_files:

                data_file.metamapped_path = os.path.join(self.metamapped_files_directory,
                                                         data_file.raw_path.split(os.path.sep)[-1]
                                                         .replace(".%s" % self.raw_text_file_extension, ".metamapped"))


    def get_data_files(self):
        """
        Retrieves an list containing all the files registered by a Dataset.
        :return: a list of DataFile objects.
        """
        return self.all_data_files[0:self.data_limit]

    def metamap(self, metamap, n_jobs=multiprocessing.cpu_count() - 1, retry_possible_corruptions=True):
        """
        Metamaps the files registered by a Dataset. Attempts to Metamap utilizing a max prune depth of 30, but on
        failure retries with lower max prune depth. A lower prune depth roughly equates to decreased MetaMap performance.
        More information can be found in the MetaMap documentation.
        :param metamap: an instance of MetaMap.
        :param n_jobs: the number of processes to spawn when metamapping. Defaults to one less core than available on your machine.
        :param retry_possible_corruptions: Re-Metamap's files that are detected as being possibly corrupt. Set to False for more control over what gets Metamapped or if you are having bugs with Metamapping. (default: True)
        :return: Inside *metamapped_files_directory* or by default inside a sub-directory of your *data_directory* named *metamapped* we have that for each raw text file, *file_name*, an auxiliary metamapped version is created and stored.
        """
        self.metamap = metamap

        if self.is_metamapped():
            return True

        #make metamap directory if it doesn't exist.
        if not os.path.isdir(self.metamapped_files_directory):
            os.makedirs(self.metamapped_files_directory)

        #A file that is below 200 bytes is likely corrupted output from MetaMap, these should be retried.
        if retry_possible_corruptions:
            #Do not metamap files that are already metamapped and above 200 bytes in size
            already_metamapped = [file[:file.find('.')] for file in os.listdir(self.metamapped_files_directory)
                                  if os.path.getsize(os.path.join(self.metamapped_files_directory, file)) > 200]
        else:
            #Do not metamap files that are already metamapped
            already_metamapped = [file[:file.find('.')] for file in os.listdir(self.metamapped_files_directory)]

        files_to_metamap = [data_file.raw_path for data_file in self.all_data_files if not data_file.file_name in already_metamapped]

        logging.info("Number of files to MetaMap: %i" % len(files_to_metamap))

        Parallel(n_jobs=n_jobs)(delayed(self._parallel_metamap)(files_to_metamap, i) for i in range(len(files_to_metamap)))

        if self.is_metamapped():
            for data_file in self.all_data_files:
                data_file.metamapped_path = os.path.join(self.metamapped_files_directory,
                                                         data_file.raw_path.split(os.path.sep)[-1]
                                                         .replace(".%s" % self.raw_text_file_extension, ".metamapped"))


    def _parallel_metamap(self, files, i):
        """
        Facilitates Metamapping in parallel by forking off processes to Metamap each file individually.
        :param files: an array of file paths to the file to map
        :param i: index in the array used to determine the file that the called process will be responsible for mapping
        :return: metamapped_files_directory now contains metamapped versions of the dataset files
        """
        file = files[i].split(os.path.sep)[-1]
        file_path = files[i]
        logging.info("Attempting to Metamap: %s", file_path)
        mapped_file_location = os.path.join(self.metamapped_files_directory, file.replace(self.raw_text_file_extension, "metamapped"))

        with open(mapped_file_location, 'w') as mapped_file:
            max_prune_depth = 30  # this is the maximum prune depth metamap utilizes when concept mapping

            metamap_dict = None
            # while current prune depth causes out of memory on document
            while metamap_dict is None or metamap_dict['metamap'] is None:
                if max_prune_depth <= 0:
                    logging.critical("Failed to to metamap after multiple attempts: %s", file_path)
                    return
                try:
                    metamap_dict = self.metamap.map_file(file_path, max_prune_depth=max_prune_depth) #attempt to metamap
                    if metamap_dict['metamap'] is not None: #if successful
                        break
                    max_prune_depth = int(math.e ** (math.log(max_prune_depth) - .5)) #decrease prune depth by an order of magnitude
                except BaseException as e:
                    metamap_dict = None
                    max_prune_depth = int(math.e ** (math.log(max_prune_depth) - .5)) #decrease prune depth by an order of magnitude
                    logging.warning("Error Metamapping: %s with exception %s", file_path, str(e))

            mapped_file.write(json.dumps(metamap_dict))
            logging.info("Successfully Metamapped: %s", file_path)
            logging.info("Successfully Metamapped: %s" % file_path)

    def is_metamapped(self):
        """
        Verifies if all fil es in the Dataset are metamapped.
        :return: True if all data files are metamapped, False otherwise.
        """
        if not os.path.isdir(self.metamapped_files_directory):
            return False

        for file in self.all_data_files:
            potential_file_path = os.path.join(self.metamapped_files_directory, "%s.metamapped" % file.file_name)
            if not os.path.isfile(potential_file_path):
                return False

            # Metamapped file could exist, but metamapping it could have failed.
            # If the file is less than 200 bytes, log a warning.
            file_size_in_bytes = os.path.getsize(potential_file_path)
            if file_size_in_bytes < 200:
                logging.warning("Metamapped version of %s is only %i bytes. Metamapping could have failed: %s" %
                                (file.file_name, file_size_in_bytes, potential_file_path))

        return True

    def is_training(self):
        """
        Whether this Dataset can be used for training.
        :return: True if training dataset, false otherwise. A training dataset is a collection raw text and corresponding annotation files while a prediction dataset contains solely raw text files.
        """
        return self.is_training_directory

    def set_data_limit(self, data_limit):
        """
        A limit to the number of files in the Dataset that medaCy works with
        This is useful for preliminary experimentation when working with an entire Dataset takes time.
        :return:
        """
        self.data_limit = data_limit

    def get_data_directory(self):
        """
        Retrieves the directory this Dataset abstracts from.
        :return:
        """
        return self.data_directory

    def __str__(self):
        """Converts self.get_data_files() to a list of strs and combines them into one str"""
        return "[%s]" % ", ".join([str(x) for x in self.get_data_files()])


    @staticmethod
    def load_external(package_name):
        """
        Loads an external medaCy compatible dataset. Requires the dataset's associated package to be installed.
        Alternatively, you can import the package directly and call it's .load() method.
        :param package_name: the package name of the dataset
        :return: A tuple containing a training set, evaluation set, and meta_data
        """
        if importlib.util.find_spec(package_name) is None:
            raise ImportError("Package not installed: %s" % package_name)
        return importlib.import_module(package_name).load()









