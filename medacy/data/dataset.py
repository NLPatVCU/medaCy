"""
A medaCy Dataset facilities the management of data for both model training and model prediction.

A Dataset object provides a wrapper for a unix file directory containing training/prediction
data. If a Dataset, at training time, is fed into a pipeline requiring auxilary files
(Metamap for instance) the Dataset will automatically create those files in the most efficient way possible.

Training
#################
When a directory contains **both** raw text files alongside annotation files, an instantiated Dataset
detects and facilitates access to those files.

Assuming your directory looks like this (where .ann files are in `BRAT <http://brat.nlplab.org/standoff.html>`_ format):
::
    home/medacy/data
    ├── file_one.ann
    ├── file_one.txt
    ├── file_two.ann
    └── file_two.txt

A common data work flow might look as follows.

Running:
::
    >>> from medacy.data import Dataset
    >>> from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap

    >>> dataset = Dataset('/home/datasets/some_dataset')
    >>> for data_file in dataset:
    ...    (data_file.file_name, data_file.raw_path, dataset.ann_path)
    (file_one, file_one.txt, file_one.ann)
    (file_two, file_two.txt, file_two.ann)
    >>> dataset
    ['file_one', 'file_two']
    >>>> dataset.is_metamapped()
    False
    >>> metamap = MetaMap('/home/path/to/metamap/binary')
    >>> with metamap:
    ...     dataset.metamap(metamap)
    >>> dataset.is_metamapped()
    True

MedaCy **does not** alter the data you load in any way - it only reads from it.

Prediction
##########
When a directory contains **only** raw text files, an instantiated Dataset object interprets this as
a directory of files that need to be predicted. This means that the internal Datafile that aggregates
meta-data for a given prediction file does not have fields for annotation_file_path set.

When a directory contains **only** ann files, an instantiated Dataset object interprets this as
a directory of files that are predictions. Useful methods for analysis include :meth:`medacy.data.dataset.Dataset.compute_confusion_matrix`,
:meth:`medacy.data.dataset.Dataset.compute_ambiguity` and :meth:`medacy.data.dataset.Dataset.compute_counts`.

External Datasets
#################

In the real world, datasets (regardless of domain) are evolving entities. Hence, it is essential to version them.
A medaCy compatible dataset can be created to facilitate this versioning. A medaCy compatible dataset lives a python
packages that can be hooked into medaCy or used for any other purpose - it is simply a loose wrapper for this Dataset
object. Instructions for creating such a dataset can be found `here <https://github.com/NLPatVCU/medaCy/tree/master/examples/guide>`_.
wrap them.
"""
import importlib
import json
import logging
import math
import multiprocessing
import os
from collections import Counter

import spacy
from joblib import Parallel, delayed

from medacy.data.annotations import Annotations
from medacy.data.data_file import DataFile


class Dataset:
    """
    A facilitation class for data management.
    """

    def __init__(self, data_directory, raw_text_file_extension="txt", metamapped_files_directory=None, data_limit=None):
        """
        Manages directory of training data along with other medaCy generated files.

        Only text files: considers a directory for managing metamapping.
        Only ann files: considers a directory of predictions.
        Both text and ann files: considers a directory for training.

        :param data_directory: Directory containing data for training or prediction.
        :param raw_text_file_extension: The file extension of raw text files in the data_directory (default: *.txt*)
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

        if not raw_text_files:  # detected a prediction directory
            ann_files = sorted([file for file in all_files_in_directory if file.endswith('.ann')])
            self.is_training_directory = False

            if data_limit is not None:
                self.data_limit = data_limit
            else:
                self.data_limit = len(ann_files)

            for file in ann_files:
                annotation_path = os.path.join(data_directory, file)
                file_name = file[:-len('.ann') - 1]
                self.all_data_files.append(DataFile(file_name, None, annotation_path))

        else:  # detected a training directory (raw text files exist)
            if data_limit is not None:
                self.data_limit = data_limit
            else:
                self.data_limit = len(raw_text_files)

            if self.data_limit < 1 or self.data_limit > len(raw_text_files):
                raise ValueError(
                    "Parameter 'data_limit' must be between 1 and number of raw text files in data_directory")

            # required ann files for this to be a training directory
            ann_files = [file.replace(".%s" % raw_text_file_extension, ".ann") for file in raw_text_files]
            # only a training directory if every text file has a corresponding ann_file
            self.is_training_directory = all(os.path.isfile(os.path.join(data_directory, ann_file)) for ann_file in ann_files)

            # set all file attributes except metamap_path as it is optional.
            for file in raw_text_files:
                file_name = file[:-len(raw_text_file_extension) - 1]
                raw_text_path = os.path.join(data_directory, file)

                if self.is_training_directory:
                    annotation_path = os.path.join(
                        data_directory,
                        file.replace(".%s" % raw_text_file_extension, '.ann')
                    )
                else:
                    annotation_path = None
                self.all_data_files.append(DataFile(file_name, raw_text_path, annotation_path))

            #If directory is already metamapped, use it.
            if self.is_metamapped():
                for data_file in self.all_data_files:
                    data_file.metamapped_path = os.path.join(
                        self.metamapped_files_directory,
                        data_file.txt_path.split(os.path.sep)[-1].replace(".%s" % self.raw_text_file_extension, ".metamapped")
                    )

    def get_data_files(self):
        """
        Retrieves an list containing all the files registered by a Dataset.

        :return: a list of DataFile objects.
        """
        return self.all_data_files[0:self.data_limit]

    def __iter__(self):
        return iter(self.get_data_files())

    def __len__(self):
        return len(self.get_data_files())

    def get_training_data(self, data_format='spacy'):
        """
        Get training data in a specified format.

        :param data_format: The specified format as a string.
        :return: The requested data in the requested format.
        """
        supported_formats = ['spacy']

        if data_format not in supported_formats:
            raise TypeError("Format %s not supported" % data_format)

        training_data = []
        nlp = spacy.load('en_core_web_sm')

        for ann in self.generate_annotations():
            training_data.append(ann.get_entity_annotations(format=data_format, nlp=nlp))

        return training_data

    def get_subdataset(self, indices):
        """
        Get a subdataset of data files based on indices.

        :param indices: List of ints that represent the indexes of the data files to split off.
        :return: Dataset object with only the specified data files.
        """
        subdataset = Dataset(self.data_directory)
        data_files = subdataset.get_data_files()
        sub_data_files = []

        for index, data_file in enumerate(data_files):
            if index in indices:
                sub_data_files.append(data_file)

        subdataset.all_data_files = sub_data_files
        return subdataset

    def metamap(self, metamap, n_jobs=multiprocessing.cpu_count() - 1, retry_possible_corruptions=True):
        """
        Metamaps the files registered by a Dataset. Attempts to Metamap utilizing a max prune depth of 30, but on
        failure retries with lower max prune depth. A lower prune depth roughly equates to decreased MetaMap performance.
        More information can be found in the MetaMap documentation.

        Example usage:
        >>> metamap = MetaMap("/path/to/metamap")
        >>> data = Dataset("/path/to/data")
        >>> with metamap:
        ...     data.metamap(metamap)

        :param metamap: an instance of MetaMap.
        :param n_jobs: the number of processes to spawn when metamapping. Defaults to one less core than available on your machine.
        :param retry_possible_corruptions: Re-Metamap's files that are detected as being possibly corrupt. Set to False for more control over what gets Metamapped or if you are having bugs with Metamapping. (default: True)
        :return: Inside *metamapped_files_directory* or by default inside a sub-directory of your *data_directory* named *metamapped* we have that for each raw text file, *file_name*, an auxiliary metamapped version is created and stored.
        """
        self.metamap = metamap

        if self.is_metamapped():
            return True

        # make metamap directory if it doesn't exist.
        if not os.path.isdir(self.metamapped_files_directory):
            os.makedirs(self.metamapped_files_directory)

        # A file that is below 200 bytes is likely corrupted output from MetaMap, these should be retried.
        if retry_possible_corruptions:
            # Do not metamap files that are already metamapped and above 200 bytes in size
            already_metamapped = [file[:file.find('.')] for file in os.listdir(self.metamapped_files_directory)
                                  if os.path.getsize(os.path.join(self.metamapped_files_directory, file)) > 200]
        else:
            # Do not metamap files that are already metamapped
            already_metamapped = [file[:file.find('.')] for file in os.listdir(self.metamapped_files_directory)]

        files_to_metamap = [data_file.txt_path for data_file in self.all_data_files if data_file.file_name not in already_metamapped]

        logging.info("Number of files to MetaMap: %i" % len(files_to_metamap))

        Parallel(n_jobs=n_jobs)(delayed(self._parallel_metamap)(files_to_metamap, i) for i in range(len(files_to_metamap)))

        if self.is_metamapped():
            for data_file in self.all_data_files:
                data_file.metamapped_path = os.path.join(self.metamapped_files_directory,
                                                         data_file.txt_path.split(os.path.sep)[-1]
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
                    logging.warning(f"Error Metamapping: {file_path} after raising {type(e).__name__}: {str(e)}")

            mapped_file.write(json.dumps(metamap_dict))
            logging.info("Successfully Metamapped: %s", file_path)

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

    def __str__(self):
        """
        Prints a list-like string of the names of the Datafile objects up to the data limit
        (can't be used if copied and pasted)
        """
        return str([d.file_name for d in self.get_data_files()])

    def compute_counts(self):
        """
        Computes entity counts over all documents in this dataset.

        :return: a Counter of entity counts
        """
        total = Counter()

        for ann in self.generate_annotations():
            total += ann.compute_counts()

        return total

    def compute_confusion_matrix(self, dataset, leniency=0):
        """
        Generates a confusion matrix where this Dataset serves as the gold standard annotations and `dataset` serves
        as the predicted annotations. A typical workflow would involve creating a Dataset object with the prediction directory
        outputted by a model and then passing it into this method.

        :param dataset: a Dataset object containing a predicted version of this dataset.
        :param leniency: a floating point value between [0,1] defining the leniency of the character spans to count as different. A value of zero considers only exact character matches while a positive value considers entities that differ by up to :code:`ceil(leniency * len(span)/2)` on either side.
        :return: two element tuple containing a label array (of entity names) and a matrix where rows are gold labels and columns are predicted labels. matrix[i][j] indicates that entities[i] in this dataset was predicted as entities[j] in 'annotation' matrix[i][j] times
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("dataset must be instance of Dataset")

        # verify files are consistent
        diff = set(file.ann_path.split(os.sep)[-1] for file in self) - set(file.ann_path.split(os.sep)[-1] for file in dataset)
        if diff:
            raise ValueError("Dataset of predictions is missing the files: "+str(list(diff)))

        #sort entities in ascending order by count.
        entities = [key for key, _ in sorted(self.compute_counts().items(), key=lambda x: x[1])]
        confusion_matrix = [[0 for x in range(len(entities))] for x in range(len(entities))]

        for gold_data_file in self:
            prediction_iter = iter(dataset)
            prediction_data_file = next(prediction_iter)
            while str(gold_data_file) != str(prediction_data_file):
                prediction_data_file = next(prediction_iter)

            gold_annotation = Annotations(gold_data_file.ann_path)
            pred_annotation = Annotations(prediction_data_file.ann_path)

            #compute matrix on the Annotation file level
            ann_confusion_matrix = gold_annotation.compute_confusion_matrix(pred_annotation, entities, leniency=leniency)
            for i in range(len(confusion_matrix)):
                for j in range(len(confusion_matrix)):
                    confusion_matrix[i][j] += ann_confusion_matrix[i][j]

        return entities, confusion_matrix

    def compute_ambiguity(self, dataset):
        """
        Finds occurrences of spans from 'dataset' that intersect with a span from this annotation but do not have this spans label.
        label. If 'dataset' comprises a models predictions, this method provides a strong indicators
        of a model's in-ability to dis-ambiguate between entities. For a full analysis, compute a confusion matrix.

        :param dataset: a Dataset object containing a predicted version of this dataset.
        :param leniency: a floating point value between [0,1] defining the leniency of the character spans to count as different. A value of zero considers only exact character matches while a positive value considers entities that differ by up to :code:`ceil(leniency * len(span)/2)` on either side.
        :return: a dictionary containing the ambiguity computations on each gold, predicted file pair
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("dataset must be instance of Dataset")

        # verify files are consistent
        diff = set(file.ann_path.split(os.sep)[-1] for file in self) - set(file.ann_path.split(os.sep)[-1] for file in dataset)
        if diff:
            raise ValueError("Dataset of predictions is missing the files: " + str(list(diff)))

        #Dictionary storing ambiguity over dataset
        ambiguity_dict = {}

        for gold_data_file in self:
            prediction_iter = iter(dataset)
            prediction_data_file = next(prediction_iter)
            while str(gold_data_file) != str(prediction_data_file):
                prediction_data_file = next(prediction_iter)

            gold_annotation = Annotations(gold_data_file.ann_path)
            pred_annotation = Annotations(prediction_data_file.ann_path)

            # compute matrix on the Annotation file level
            ambiguity_dict[str(gold_data_file)] = gold_annotation.compute_ambiguity(pred_annotation)


        return ambiguity_dict

    @staticmethod
    def load_external(package_name):
        """
        Loads an external medaCy compatible dataset. Requires the dataset's associated package to be installed.
        Alternatively, you can import the package directly and call it's .load() method.

        :param package_name: the package name of the dataset
        :return: A tuple containing a training set, evaluation set
        """
        if importlib.util.find_spec(package_name) is None:
            raise ImportError("Package not installed: %s" % package_name)
        return importlib.import_module(package_name).load()

    def get_labels(self, as_list=False):
        """
        Get all of the entities/labels used in the dataset.
        :param as_list: bool for if to return the results as a list; defaults to False
        :return: A set of strings. Each string is a label used.
        """
        labels = set()

        for ann in self.generate_annotations():
            labels.update(ann.get_labels())

        if as_list:
            return list(labels)
        return labels

    def generate_annotations(self):
        """Generates Annotation objects for all the files in this Dataset"""
        for file in self.get_data_files():
            yield Annotations(file.ann_path, source_text_path=file.txt_path)

    def __getitem__(self, item):
        """
        Creates and returns the Annotations object with the given file name, else raises FileNotFoundError;
        useful for getting Annotations objects from parallel Datasets
        :param item: the name of the file to be represented (not including the extension or parent directories)
        :return: an Annotations object
        """
        path = os.path.join(self.data_directory, item, '.ann')
        return Annotations(path)
