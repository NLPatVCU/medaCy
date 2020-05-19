"""
Inter-dataset agreement calculator

This module calculates precision, recall, and F1 scores given two parallel datasets with a strict or lenient setting.
The strict setting will only count true positives from the predicted data if they have an exact match, span for span,
with the same label in the gold dataset. Lenient results count at most one true positive per named entity in the gold
dataset, so if more than one entity in the predicted data is a lenient match to a given entity in the gold data, only
the first match counts towards the true positive score. However, subsequent lenient matches to a gold entity that has
already been paired will not count as false positives.
"""

import argparse
import logging
from collections import OrderedDict
from itertools import product
from statistics import mean

from tabulate import tabulate

from medacy.data.dataset import Dataset
from medacy.tools.entity import Entity


class Measures:
    """
    Data type for binary classification scores scores

    :ivar tp: A number of true positives
    :ivar fp: A number of false positives
    :ivar tn: A number of true negatives
    :ivar fn: A number of false negatives
    """

    def __init__(self, tp=0, fp=0, tn=0, fn=0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def __eq__(self, other):
        return (self.tp, self.fp, self.tn, self.fn) == (other.tp, other.fp, other.tn, other.fn)

    def __repr__(self):
        return f"{type(self).__name__}(tp={self.tp}, fp={self.fp}, tn={self.tn}, fn={self.fn})"

    def __add__(self, other):
        tp = self.tp + other.tp
        fp = self.fp + other.fp
        tn = self.tn + other.tn
        fn = self.fn + other.fn
        return Measures(tp=tp, fp=fp, tn=tn, fn=fn)

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn
        return self

    def precision(self):
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1):
        """Compute F score given a custom beta"""
        if beta <= 0:
            raise ValueError("beta must be >= 0")
        prec = self.precision()
        rec = self.recall()
        num = (1 + beta ** 2) * (prec * rec)
        den = beta ** 2 * (prec + rec)
        try:
            return num / den
        except ZeroDivisionError:
            return 0.0

    def specificity(self):
        """Compute Specificity score."""
        try:
            return self.tn / (self.fp + self.tn)
        except ZeroDivisionError:
            return 0.0

    def sensitivity(self):
        """Compute Sensitivity score."""
        return self.recall()

    def auc(self):
        """Compute AUC score."""
        return (self.sensitivity() + self.specificity()) / 2

    def accuracy(self):
        try:
            return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        except ZeroDivisionError:
            return 0.0


def zip_datasets(dataset_1, dataset_2):
    """
    Takes two Datasets, determines how much overlap there is between them, and returns pairs of the matching ann files
    :param dataset_1: The first Dataset
    :param dataset_2: The second Dataset
    :return: an iterator of zipped ann file tuples, with files from dataset_1 first and dataset_2 second
    """
    dataset_1_only = {d.file_name for d in dataset_1} - {d.file_name for d in dataset_2}
    if dataset_1_only:
        logging.warning(f"The following files only appear in {dataset_1.data_directory}: {dataset_1_only}")

    dataset_2_only = {d.file_name for d in dataset_2} - {d.file_name for d in dataset_1}
    if dataset_2_only:
        logging.warning(f"The following files only appear in {dataset_2.data_directory}: {dataset_2_only}")

    matching_file_names = {d.file_name for d in dataset_1} & {d.file_name for d in dataset_2}
    dataset_1_ann_files = [d.ann_path for d in dataset_1 if d.file_name in matching_file_names]
    dataset_1_ann_files.sort()
    dataset_2_ann_files = [d.ann_path for d in dataset_2 if d.file_name in matching_file_names]
    dataset_2_ann_files.sort()

    yield from zip(dataset_1_ann_files, dataset_2_ann_files)


def measure_ann_file(ann_1, ann_2, mode='strict'):
    """
    Calculates tag level measurements for two parallel ann files; it does not score them
    :param ann_1: path to the gold ann file
    :param ann_2: path to the system ann file
    :param mode: strict or lenient
    :return: a dictionary mapping tags (str) to measurements (Measures)
    """
    if mode not in ['strict', 'lenient']:
        raise ValueError("mode must be 'strict' or 'lenient'")

    gold_ents = Entity.init_from_doc(ann_1)
    system_ents = Entity.init_from_doc(ann_2)

    unmatched_gold = gold_ents.copy()
    unmatched_system = system_ents.copy()

    # While we're only interested in tags used by the gold dataset, these tags are calculated
    # at the document level, and it's possible that a tag that appears somewhere else in the gold
    # dataset does not appear in this gold file, but is still predicted for by mistake; thus we use both
    tags = {e.tag for e in gold_ents} | {e.tag for e in system_ents}
    measures = {tag: Measures() for tag in tags}

    for s, g in product(system_ents, gold_ents):
        if s.equals(g, mode=mode):
            if s not in unmatched_system:
                # Don't do anything with system predictions that have already been paired
                continue

            if g in unmatched_gold:
                # Each gold entity can only be matched to one prediction and
                # can only count towards the true positive score once
                unmatched_gold.remove(g)
                unmatched_system.remove(s)
                measures[s.tag].tp += 1
            else:
                # The entity has been matched to a gold entity, but we have
                # already gotten the one true positive match allowed for each gold entity;
                # therefore we say that the predicted entity is now matched
                unmatched_system.remove(s)

    for s in unmatched_system:
        # All predictions that don't match any gold entity count one towards the false positive score
        measures[s.tag].fp += 1

    for tag, measure in measures.items():
        # The number of false negatives is the number of gold entities for a tag minus the number that got
        # counted as true positives
        measures[tag].fn = len([e for e in gold_ents if e.tag == tag]) - measure.tp

    return measures


def measure_dataset(gold_dataset, system_dataset, mode='strict'):
    """
    Measures the true positive, false positive, and false negative counts for a directory of predictions
    :param gold_dataset: The gold version of the predicted dataset
    :param system_dataset: The predicted dataset
    :param mode: 'strict' or 'lenient'
    :return: a dictionary of tag-level Measures objects
    """
    if mode not in ['strict', 'lenient']:
        raise ValueError("mode must be 'strict' or 'lenient'")

    all_file_measures = []
    tag_measures = {tag: Measures() for tag in gold_dataset.get_labels()}

    for gold, system in zip_datasets(gold_dataset, system_dataset):
        all_file_measures.append(measure_ann_file(gold, system, mode=mode))

    # Combine the Measures objects for each tag from each file together
    for file_measures in all_file_measures:
        for tag, measure in file_measures.items():
            tag_measures[tag] += measure

    return tag_measures


def format_results(measures_dict, num_dec=3, table_format='plain'):
    """
    Runs calculations on Measures objects and returns a printable table (but does not print it)
    :param measures_dict: A dictionary mapping tags (str) to Measures
    :param num_dec: number of decimal places to round to
    :param table_format: a tabulate module table format (see tabulate on PyPI)
    :return: a string of tabular data
    """
    # Alphabetize the dictionary
    measures_dict = OrderedDict(sorted(measures_dict.items()))

    table = [['Tag', 'Prec', 'Rec', 'F1']]

    for tag, m in measures_dict.items():
        table.append([
            tag,
            m.precision(),
            m.recall(),
            m.f_score()
        ])

    table.append([
        'system (macro)',
        mean(m.precision() for m in measures_dict.values()),
        mean(m.recall() for m in measures_dict.values()),
        mean(m.f_score() for m in measures_dict.values())
    ])

    combined_measures = sum(measures_dict.values(), Measures())

    table.append([
        'system (micro)',
        combined_measures.precision(),
        combined_measures.recall(),
        combined_measures.f_score()
    ])

    return tabulate(table, headers='firstrow', tablefmt=table_format, floatfmt=f".{num_dec}f")


def main():
    parser = argparse.ArgumentParser(description='Inter-dataset agreement calculator')
    parser.add_argument('gold_directory', help='First data folder path (gold)')
    parser.add_argument('system_directory', help='Second data folder path (system)')
    parser.add_argument('-m', '--mode', default='strict', help='strict or lenient (defaults to strict)')
    parser.add_argument('-f', '--format', default='plain', help='format to print the table (options include grid, github, and latex)')
    parser.add_argument('-d', '--decimal', type=int, default=3, help='number of decimal places to round to')
    args = parser.parse_args()

    gold_dataset = Dataset(args.gold_directory)
    system_dataset = Dataset(args.system_directory)

    result = measure_dataset(gold_dataset, system_dataset, args.mode)
    output = format_results(result, num_dec=args.decimal, table_format=args.format)
    print(output)


if __name__ == '__main__':
    main()
