import os
import copy
from medacy.data.dataset import Dataset
from medacy.tools.entity import Entity
from statistics import mean

def calculate_inter_dataset_agreement(predicted_dataset: Dataset, gold_dataset: Dataset, different_entity_mappings: [tuple]):
    """
    Given a gold dataset and a dataset of predictions on that dataset, calculate the precision, recall, and f1 of the
    predicted dataset.
    :param predicted_dataset: The dataset to be evaluated
    :param gold_dataset: The gold dataset
    :param different_entity_mappings: A mapping of entity class names that are not the same in each dataset given as a
    list of tuples, where the first value in the tuple is name of that class in the predicted dataset and the
    second value is the name of that class in the gold dataset.
    :return: The tab-delimited values
    """
    # Get the list of entities in each dataset
    predicted_entities = predicted_dataset.get_labels()
    gold_entities = gold_dataset.get_labels()

    # Define the default mappings, which will be any time the same entity type appears in both datasets
    matching_entities = predicted_entities.intersection(gold_entities)
    entity_mappings = {e: e for e in matching_entities}

    # Get the entity counts in the gold dataset
    gold_entity_counts = gold_dataset.compute_counts()
    predicted_entity_counts = predicted_dataset.compute_counts()

    # Create the data dictionary
    entity_dict = {
        "ti": 0,  # total instances in the gold dataset
        "tp": 0,  # true positives in the predicted set
        "ta": 0  # total attempts; ie sum of true and false positives
    }

    data_dict = {e: copy.copy(entity_dict) for e in matching_entities}

    for item in different_entity_mappings:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("different_entity_mappings must be a list of doubles (2-tuples) only")

    for entity in predicted_entities:
        if entity not in gold_entities:
            if entity not in [f[0] for f in different_entity_mappings]:
                raise ValueError("Entity classes that do not appear in both datasets must have a mapping given, "
                                 "but '%s' was not given a mapping" % entity)
            for i, e in enumerate([f[0] for f in different_entity_mappings]):
                # Iterate over the predicted entity names until you get to the right one,
                # then map that entity to the one its paired to in its tuple
                if e == entity:
                    mapped_entity = entity_mappings[entity] = different_entity_mappings[i][1]
                    data_dict[entity] = copy.copy(entity_dict)
                    # Map the total possible instances of that predicted entity in the gold dataset
                    # to the number of occurrences of the mapped entity in the gold dataset
                    data_dict[entity]["ti"] = gold_entity_counts[mapped_entity]
                    break  # there will always be a match, else error is raised earlier
        else:
            # If the entity type appears in both datasets with the same name, set the total attempts
            # to the number of occurrences in the predicted dataset
            data_dict[entity]["ta"] = predicted_entity_counts[entity]

    # Zip the two datasets into a single list
    for file in predicted_dataset:
        if len(predicted_dataset) != len(gold_dataset):
            raise ValueError("Both datasets must contain parallel copies of the same files, but the two datasets "
                             "are different lengths")
        if file.file_name not in gold_dataset:
            raise ValueError("File '%s' exists in the predicted dataset, but not in the gold dataset" % file.file_name)

    predicted_data_files = sorted([os.path.join(predicted_dataset.data_directory, f.file_name) for f in predicted_dataset])
    gold_data_files = sorted([os.path.join(gold_dataset.data_directory, f.file_name) for f in gold_dataset])
    all_file_pairs = zip(predicted_data_files, gold_data_files)

    for predicted, gold in all_file_pairs:
        predicted_entities = Entity.init_from_doc(predicted)
        gold_entities = Entity.init_from_doc(gold)

        for instance in predicted_entities:
            for possible_match in gold_entities:
                if instance == possible_match and instance.ent_type == entity_mappings[instance.ent_type]:
                    data_dict[instance.ent_type]["tp"] += 1
                    break

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

        if entity == entity_mappings[entity]:
            output_str += f"{entity}\t{precision}\t{recall}\t{f1}\n"
        else:
            output_str += f"{entity}->{entity_mappings[entity]}\t{precision}\t{recall}\t{f1}\n"

    system_precision = mean(all_precision)
    system_recall = mean(all_recall)
    system_f1 = mean(all_f1)

    output_str += f"system\t{system_precision}\t{system_recall}\t{system_f1}\n"

    print(output_str)
    return output_str
