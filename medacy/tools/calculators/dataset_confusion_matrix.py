import argparse
import tabulate

import numpy as np

from medacy.data.dataset import Dataset


def calculate_dataset_confusion_matrix(dataset_1_path, dataset_2_path, data_limit=None, unmatched=False, leniency=0.0):
    dataset_1 = Dataset(dataset_1_path, data_limit=data_limit)
    dataset_2 = Dataset(dataset_2_path, data_limit=data_limit)
    ents, mat = dataset_1.compute_confusion_matrix(dataset_2, unmatched=unmatched, leniency=leniency)
    return ents, mat

def format_headers(entities):
    result = []
    for string in entities:
        vertical_string = ""
        for char in string:
            vertical_string += (char+"\n")
        result.append(vertical_string.strip())
    return result

def format_density(mat):  
    for row_index, row in enumerate(mat):
        row_total = round(float(sum(row)), 3)
        if row_total==0.0:
            for col_index, item in enumerate(row):
                mat[row_index][col_index] = 0.0
        else:
            for col_index, item in enumerate(row):
                mat[row_index][col_index] /= row_total
    return mat

def main():
    parser = argparse.ArgumentParser(description="Calculate and display the ambiguity of two datasets")
    parser.add_argument('dataset_1', type=str, help="The first dataset path")
    parser.add_argument('dataset_2', type=str, help="The second dataset path")
    parser.add_argument('-dl', '--data_limit', type=int, default=None, help="The data limit to be used")
    parser.add_argument('-d', '--density', action='store_true', help="Displays values as density")
    parser.add_argument('-u', '--unmatched', action='store_true', help="Displays unmatched counts")
    parser.add_argument('-l', '--leniency', type=float, default=0.0, help="Leniency between 0.0 and 1.0 (default to 0.0)")
    args = parser.parse_args()

    ents, mat = calculate_dataset_confusion_matrix(args.dataset_1, args.dataset_2, unmatched=args.unmatched, data_limit=args.data_limit, leniency=args.leniency)

    if args.density:
        mat = format_density(mat)
    if args.unmatched:
        print(tabulate.tabulate(mat, headers=format_headers(ents + ["Unmatched"]), showindex=ents, tablefmt="plain"))
    else:
        print(tabulate.tabulate(mat, headers=format_headers(ents), showindex=ents, tablefmt="plain"))

if __name__ == '__main__':
    main()
