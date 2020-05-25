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
    ...     metamap.metamap_dataset(dataset)
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

import argparse
import json
import logging
import os
import pprint
from collections import Counter
from pathlib import Path

from medacy.data.annotations import Annotations
from medacy.data.data_file import DataFile


class Dataset:
    """
    A facilitation class for data management.
    """

    def __init__(self, data_directory, data_limit=None):
        """
        Manages directory of training data along with other medaCy generated files.

        Only text files: considers a directory for managing metamapping.
        Only ann files: considers a directory of predictions.
        Both text and ann files: considers a directory for training.

        :param data_directory: Directory containing data for training or prediction.
        :param data_limit: A limit to the number of files to process. Must be between 1 and number of raw text files in data_directory
        """
        self.data_directory = Path(data_directory)

        metamap_dir = self.data_directory / 'metamapped'
        self.metamapped_files_directory = metamap_dir if metamap_dir.is_dir() else None

        self.data_files = self._create_data_files()
        self.data_limit = data_limit or len(self.data_files)

    def _create_data_files(self):
        data_files = []
        all_files_in_directory = os.listdir(self.data_directory)
        all_file_base_names = {f.split(".")[0] for f in all_files_in_directory}

        for file_name in all_file_base_names:
            txt_path = None
            ann_path = None
            metamapped_path = None

            potential_txt_path = self.data_directory / (file_name + ".txt")
            if potential_txt_path.exists():
                txt_path = potential_txt_path

            potential_ann_path = self.data_directory / (file_name + ".ann")
            if potential_ann_path.exists():
                ann_path = potential_ann_path

            if self.metamapped_files_directory:
                potential_mm_path = self.metamapped_files_directory / (file_name + ".metamapped")
                if potential_mm_path.exists():
                    metamapped_path = potential_mm_path

            if txt_path or ann_path:
                new_df = DataFile(file_name, txt_path, ann_path, metamapped_path)
                data_files.append(new_df)

        return sorted(data_files, key=lambda x: x.file_name)

    def __iter__(self):
        return iter(self.data_files[0:self.data_limit])

    def __len__(self):
        return len(self.data_files)

    def is_metamapped(self):
        """
        Verifies if all fil es in the Dataset are metamapped.

        :return: True if all data files are metamapped, False otherwise.
        """
        if self.metamapped_files_directory is None or not self.metamapped_files_directory.exists():
            return False

        for file in self.data_files:
            potential_file_path = self.metamapped_files_directory / f"{file.file_name}.metamapped"
            if not potential_file_path.exists():
                return False

            # Metamapped file could exist, but metamapping it could have failed.
            # If the file is less than 200 bytes, log a warning.
            file_size_in_bytes = os.path.getsize(potential_file_path)
            if file_size_in_bytes < 200:
                logging.warning(f"Metamapped version of {file.file_name} is only {file_size_in_bytes} bytes. "
                                f"Metamapping could have failed: {potential_file_path}")

        return True

    def __str__(self):
        """
        Prints a list-like string of the names of the Datafile objects up to the data limit
        (can't be used if copied and pasted)
        """
        return str([d.file_name for d in self])

    def compute_counts(self):
        """
        Computes entity counts over all documents in this dataset.

        :return: a Counter of entity counts
        """
        total = Counter()

        for ann in self.generate_annotations():
            total += ann.compute_counts()

        return total

    def compute_confusion_matrix(self, other, leniency=0):
        """
        Generates a confusion matrix where this Dataset serves as the gold standard annotations and `dataset` serves
        as the predicted annotations. A typical workflow would involve creating a Dataset object with the prediction directory
        outputted by a model and then passing it into this method.

        :param other: a Dataset object containing a predicted version of this dataset.
        :param leniency: a floating point value between [0,1] defining the leniency of the character spans to count as different. A value of zero considers only exact character matches while a positive value considers entities that differ by up to :code:`ceil(leniency * len(span)/2)` on either side.
        :return: two element tuple containing a label array (of entity names) and a matrix where rows are gold labels and columns are predicted labels. matrix[i][j] indicates that entities[i] in this dataset was predicted as entities[j] in 'annotation' matrix[i][j] times
        """
        if not isinstance(other, Dataset):
            raise ValueError("other must be instance of Dataset")

        # verify files are consistent
        diff = {d.file_name for d in self} - {d.file_name for d in other}
        if diff:
            raise ValueError(f"Dataset of predictions is missing the files: {repr(diff)}")

        # sort entities in ascending order by count.
        entities = [key for key, _ in sorted(self.compute_counts().items(), key=lambda x: x[1])]
        confusion_matrix = [[0 * len(entities)] * len(entities)]

        for gold_data_file in self:
            prediction_iter = iter(other)
            prediction_data_file = next(prediction_iter)
            while str(gold_data_file) != str(prediction_data_file):
                prediction_data_file = next(prediction_iter)

            gold_annotation = Annotations(gold_data_file.ann_path)
            pred_annotation = Annotations(prediction_data_file.ann_path)

            # compute matrix on the Annotation file level
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
        :return: a dictionary containing the ambiguity computations on each gold, predicted file pair
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("dataset must be instance of Dataset")

        # verify files are consistent
        diff = {d.file_name for d in self} - {d.file_name for d in dataset}
        if diff:
            raise ValueError(f"Dataset of predictions is missing the files: {repr(diff)}")

        # Dictionary storing ambiguity over dataset
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
        for file in self:
            if file.ann_path is not None:
                yield Annotations(file.ann_path, source_text_path=file.txt_path)
            else:
                yield Annotations([])

    def __getitem__(self, item):
        """
        Creates and returns the Annotations object with the given file name, else raises FileNotFoundError;
        useful for getting Annotations objects from parallel Datasets
        :param item: the name of the file to be represented (not including the extension or parent directories)
        :return: an Annotations object
        """
        path = os.path.join(self.data_directory, item + '.ann')
        return Annotations(path)


def main():
    """CLI for retrieving dataset information"""
    parser = argparse.ArgumentParser(description='Calculate data about a given data directory')
    parser.add_argument('directory')
    args = parser.parse_args()

    dataset = Dataset(args.directory)

    entities = json.dumps(dataset.get_labels(as_list=True))
    counts = dataset.compute_counts()

    print(f"Entities: {entities}")
    pprint.pprint(counts)


if __name__ == '__main__':
    main()
