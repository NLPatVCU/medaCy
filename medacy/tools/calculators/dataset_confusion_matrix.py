import argparse
import tabulate

import numpy as np

from medacy.data.dataset import Dataset


def calculate_dataset_confusion_matrix(dataset_1_path, dataset_2_path, dl=None, leniency=0.0):
    dataset_1 = Dataset(dataset_1_path, data_limit=dl)
    dataset_2 = Dataset(dataset_2_path, data_limit=dl)
    ents, mat = dataset_1.compute_confusion_matrix(dataset_2, leniency=leniency)
    return ents, mat

def format_dataset_confusion_matrix(ents, mat):
    return tabulate.tabulate(mat, headers=format_headers(ents), showindex=ents, tablefmt="orgtbl")

def format_headers(entities):
    result = []
    for string in entities:
        vertical_string = ""
        for char in string:
            vertical_string += (char+"\n")
        result.append(vertical_string.strip())
    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate and display the ambiguity of two datasets")
    parser.add_argument('dataset_1', type=str, help="The first dataset path")
    parser.add_argument('dataset_2', type=str, help="The second dataset path")
    parser.add_argument('data_limit', type=int, help="The data limit to be used")
    parser.add_argument('-l', '--leniency', type=float, default=0.0, help="Leniency between 0.0 and 1.0 (default to 0.0)")
    args = parser.parse_args()
    ents, mat = calculate_dataset_confusion_matrix(args.dataset_1, args.dataset_2, dl=data_limit leniency=args.leniency)

    # Default
    print(format_dataset_confusion_matrix(ents, np_mat))

    # Red
    # print('\033[91m' + format_dataset_confusion_matrix(ents, mat) + '\033[0m')

if __name__ == '__main__':
    main()
