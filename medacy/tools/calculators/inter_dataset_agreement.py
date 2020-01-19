"""
Inter-dataset agreement calculator
"""


import argparse
from collections import OrderedDict

from tabulate import tabulate

from medacy.data.dataset import Dataset
from medacy.tools.entity import Entity


class Measures:
    """Data type for agreement scores"""

    def __init__(self, tp=0, fp=0, tn=0, fn=0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

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
        """Compute F1-measure score."""
        if beta <= 0:
            raise ValueError("beta must be >= 0")
        try:
            num = (1 + beta ** 2) * (self.precision() * self.recall())
            den = beta ** 2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self):
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)

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

    assert len(dataset_1_ann_files) == len(dataset_2_ann_files)

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

    tags = {e.tag for e in gold_ents} | {e.tag for e in system_ents}
    measures = {tag: Measures() for tag in tags}

    for s in system_ents:
        measure = measures[s.tag]
        for g in gold_ents:
            if s.equals(g, mode=mode):
                if s not in unmatched_system:
                    continue
                if g in unmatched_gold:
                    unmatched_gold.remove(g)
                    unmatched_system.remove(s)
                    measure.tp += 1
                else:
                    unmatched_system.remove(s)

    for s in system_ents:
        measure = measures[s.tag]
        if s in unmatched_system:
            measure.fp += 1

    for tag, measure in measures.items():
        measures[tag].fn = len([e for e in gold_ents if e.tag == tag]) - measure.tp

    return measures


def measure_dataset(gold_dataset, system_dataset, mode='strict'):
    if mode not in ['strict', 'lenient']:
        raise ValueError("mode must be 'strict' or 'lenient'")

    all_file_measures = []
    tag_measures = {tag: Measures() for tag in system_dataset.get_labels()}
    system_measures = Measures()

    for gold, system in zip_datasets(gold_dataset, system_dataset):
        all_file_measures.append(measure_ann_file(gold, system, mode=mode))

    for file_measures in all_file_measures:
        for tag, measure in file_measures.items():
            tag_measures[tag] += measure

    tag_measures['system'] = sum(tag_measures.values(), system_measures)

    return tag_measures


def _format_results(measures_dict, num_files):

    # Alphabetize the dictionary, keeping system at the end
    system_measures = measures_dict['system']
    del measures_dict['system']
    measures_dict = OrderedDict(sorted(measures_dict.items(), key=lambda t: t[0]))
    measures_dict['system'] = system_measures

    table = [
        ['Tag', 'Prec', 'Rec', 'F1']  # , 'Prec (M)', 'Rec (M)', 'F1 (M)', 'Prec (µ)', 'Rec (µ)', 'F1 (µ)'
    ]

    for tag, m in measures_dict.items():
        table.append([
            tag,
            m.precision(),
            m.recall(),
            m.f1()
        ])

    return tabulate(table)


def main():
    parser = argparse.ArgumentParser(description='Inter-dataset agreement calculator')
    parser.add_argument('gold_directory', help='First data folder path (gold)')
    parser.add_argument('system_directory', help='Second data folder path (system)')
    parser.add_argument('-m', '--mode', default='strict', help='strict or lenient (defaults to strict)')
    args = parser.parse_args()

    gold_dataset = Dataset(args.gold_directory)
    system_dataset = Dataset(args.system_directory)

    num_files = len({d.file_name for d in gold_dataset} & {d.file_name for d in system_dataset})

    result = measure_dataset(gold_dataset, system_dataset, args.mode)
    output = _format_results(result, num_files)
    print(output)


if __name__ == '__main__':
    main()
