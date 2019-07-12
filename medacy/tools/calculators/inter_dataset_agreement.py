import os
import copy
import itertools
from medacy.data.dataset import Dataset
from medacy.tools.entity import Entity
from statistics import mean
from pprint import pprint

def calculate_inter_dataset_agreement(predicted_dataset: Dataset, gold_dataset: Dataset, different_entity_mappings: dict):
    """
    Given a gold dataset and a dataset of predictions on that dataset, calculate the precision, recall, and f1 of the
    predicted dataset.
    :param predicted_dataset: The dataset to be evaluated
    :param gold_dataset: The gold dataset
    :param different_entity_mappings: A mapping of entity class names that are not the same in each dataset given as a
    list of tuples, where the first value in the tuple is name of that class in the predicted dataset and the
    second value is the name of that class in the gold dataset. If there is more than one name in the gold dataset,
    give the second value in the tuple as another tuple
    :return: The tab-delimited values
    """
    if isinstance(predicted_dataset, str) and os.path.isdir(predicted_dataset):
        predicted_dataset = Dataset(predicted_dataset)
    elif not isinstance(predicted_dataset, Dataset):
        raise TypeError("predicted_dataset must be type Dataset or a str referring to a valid directory")

    if isinstance(gold_dataset, str) and os.path.isdir(gold_dataset):
        gold_dataset = Dataset(gold_dataset)
    elif not isinstance(gold_dataset, Dataset):
        raise TypeError("gold_dataset must be type Dataset or a str referring to a valid directory")

    if len(predicted_dataset) != len(gold_dataset):
        raise ValueError("Both datasets must contain parallel copies of the same files, but the two datasets "
                         "are different lengths")

    # Create the mapping dictionaries
    gold_to_pred = different_entity_mappings
    pred_to_gold = {}
    for k, v in gold_to_pred.items():
        if isinstance(v, str):
            pred_to_gold[v] = k
        elif isinstance(v, tuple):
            for w in v:
                pred_to_gold[w] = k

    # pprint(pred_to_gold)

    predicted_entities = predicted_dataset.get_labels()
    gold_entities = gold_dataset.get_labels()

    # Map the entities that are the same in both datasets to themselves;
    # if the entity exists in both datasets but is mapped, don't map it to itself
    mapped_in_gold = set(itertools.chain(*gold_to_pred.values()))
    mapped_in_pred = gold_to_pred.values()
    unmapped_in_gold = gold_entities.difference(mapped_in_gold)
    unmapped_in_pred = predicted_entities.difference(mapped_in_pred)

    same_in_both = unmapped_in_gold.intersection(unmapped_in_pred)

    # Create the data structure that will be added to
    entity_dict = {
        "ti": 0,  # total instances in the gold dataset
        "tp": 0,  # true positives in the predicted set
        "ta": 0  # total attempts; ie sum of true and false positives
    }
    data_dict = {k: copy.copy(entity_dict) for k in same_in_both}
    # pprint(data_dict); exit()
    pprint(mapped_in_pred); exit()

    for k in mapped_in_pred:
        data_dict[k] = copy.copy(entity_dict)


    # pprint(data_dict)

    # Zip the two datasets into a single list
    # for file in predicted_dataset:
    #     if file.file_name not in gold_dataset:
    #         raise ValueError("File '%s' exists in the predicted dataset, but not in the gold dataset" % file.file_name)

    predicted_data_files = sorted([os.path.join(predicted_dataset.data_directory, f.file_name + ".ann") for f in predicted_dataset])
    gold_data_files = sorted([os.path.join(gold_dataset.data_directory, f.file_name + ".ann") for f in gold_dataset])
    all_file_pairs = zip(predicted_data_files, gold_data_files)

    for predicted, gold in all_file_pairs:
        predicted_entities = Entity.init_from_doc(predicted)
        gold_entities = Entity.init_from_doc(gold)

        for instance in predicted_entities:
            # if instance.ent_type not in data_dict.keys():
            #     continue
            data_dict[instance.ent_type]["ta"] += 1
            for possible_match in gold_entities:
                if instance == possible_match and possible_match.ent_type == pred_to_gold[instance.ent_type]:
                    data_dict[instance.ent_type]["tp"] += 1

        for instance in gold_entities:
            if instance.ent_type in data_dict.keys():
                data_dict[instance.ent_type]["ti"] += 1

    all_precision = []
    all_recall = []
    all_f1 = []

    output_str = "Entity_Type\tPrecision\tRecall\tF1\n"

    for entity, values in data_dict.items():
        try: precision = values["tp"] / values["ta"]
        except ZeroDivisionError: precision = 0
        all_precision.append(precision)

        try: recall = values["tp"] / values["ti"]
        except ZeroDivisionError: recall = 0
        all_recall.append(recall)

        try: f1 = 2 * ((precision * recall)/(precision + recall))
        except ZeroDivisionError: f1 = 0
        all_f1.append(f1)

        output_str += f"{entity}\t{precision}\t{recall}\t{f1}\n"

    system_precision = mean(all_precision)
    system_recall = mean(all_recall)
    system_f1 = mean(all_f1)

    output_str += f"system\t{system_precision}\t{system_recall}\t{system_f1}\n"

    print(output_str)
    return output_str
