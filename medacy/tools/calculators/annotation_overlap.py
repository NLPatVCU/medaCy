import argparse
from collections import Counter
from itertools import product
from pprint import pprint

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset


def calculate_document_overlap(data_file):
    already_matched = []

    print(data_file.txt_path)
    ann = Annotations(data_file.ann_path)
    counts = Counter()

    for a, b in product(ann, ann):

        if a is b or {a, b} in already_matched:
            continue

        already_matched.append({a, b})

        a_tag, a_start, a_end, a_text = a
        b_tag, b_start, b_end, b_text = b

        left_cut = a_start < b_start < a_end < b_end
        right_cut = b_start < a_start < b_end < a_end
        a_inside = b_start < a_start < a_end < b_end
        b_inside = a_start < b_start < b_end < a_end

        if left_cut:
            print(f"Leftside cutoff: {a}, {b}")
        elif right_cut:
            print(f"Rightside cutoff: {a}, {b}")
        elif a_inside:
            print(f"A inside B: {a}, {b}")
        elif b_inside:
            print(f"B inside A: {a}, {b}")

        if any([left_cut, right_cut, a_inside, b_inside]):
            counts[(a_tag, b_tag)] += 1

    print(counts)
    return counts


def calculate_dataset_overlap(dataset):
    total_counts = Counter()
    for d in dataset:
        total_counts += calculate_document_overlap(d)

    print(f"Total overlaps:")
    pprint(total_counts)


def main():
    parser = argparse.ArgumentParser(description="Display which annotations in a dataset overlap")
    parser.add_argument("dataset", help="Directory of the dataset")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    calculate_dataset_overlap(dataset)


if __name__ == '__main__':
    main()

