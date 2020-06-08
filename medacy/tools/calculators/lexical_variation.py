"""
A command-line tool for creating tabular data regarding the lexical variation
of a given dataset.

python -m medacy.tools.calculators.lexical_variation --help

The output of this tool is compatible with the tabulate module
"""

import argparse

import tabulate

from medacy.data.dataset import Dataset


def calculate_unique_mentions(dataset):
    """
    Creates a dictionary of sets of unique mentions for each tag in a dataset
    :param dataset: A Dataset object
    :return: A dictionary mapping tags (str) to a set of mentions (set)
    """
    labels = dataset.get_labels()
    unique_mentions = {t: set() for t in labels}

    for ann in dataset.generate_annotations():
        for ent in ann:
            tag, start, end, text = ent
            unique_mentions[tag].add(text)

    return unique_mentions


def main():
    parser = argparse.ArgumentParser(description="Calculate the lexical variation in a given dataset")
    parser.add_argument('dataset', help="Path to the dataset directory")
    parser.add_argument('-f', '--format', help="Format to print the table (options include grid, github, and latex)")
    args = parser.parse_args()

    data = Dataset(args.dataset)
    unique_mention_dict = calculate_unique_mentions(data)
    tag_counts = data.compute_counts()

    table = [['Tag', 'Unique Mentions', 'Total Mentions', 'Ratio']]
    for tag, mentions in unique_mention_dict.items():
        table.append([tag, len(mentions), tag_counts[tag], len(mentions) / tag_counts[tag]])

    print(tabulate.tabulate(table, headers="firstrow", tablefmt=args.format))


if __name__ == '__main__':
    main()
