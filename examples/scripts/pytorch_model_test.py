"""
Script for training Pytorch models.
"""
from __future__ import unicode_literals, print_function

from pathlib import Path
from datetime import datetime
import time
import logging
import plac
from medacy.data import Dataset
from medacy.ner import Model
from medacy.ner.pipelines import LstmClinicalPipeline

@plac.annotations(
    input_dir=("Directory of ann and txt files.", "option", "i", Path),
    folds=("Folds for cross validation", "option", "f", int),
)
def main(input_dir, folds=5):
    """Main function."""
    logging.basicConfig(filename='pytorch.log', level=logging.INFO)
    # logging.basicConfig(level=logging.INFO)

    start_time = time.time()
    current_time = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('\nSTART TIME: ' + current_time)

    dataset = Dataset(input_dir)
    labels = list(dataset.get_labels())
    pipeline = LstmClinicalPipeline(entities=labels)

    model = Model(pipeline)
    model.cross_validate(num_folds=5, training_dataset=dataset)
    
    end_time = time.time()
    current_time = datetime.fromtimestamp(end_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('END TIME: ' + current_time)

    seconds_elapsed = end_time - start_time
    minutes_elapsed = seconds_elapsed / 60.0
    hours_elapsed = minutes_elapsed / 60.0
    logging.info('HOURS ELAPSED: %.0f:%.02f' % (hours_elapsed, minutes_elapsed))

if __name__ == "__main__":
    plac.call(main)
