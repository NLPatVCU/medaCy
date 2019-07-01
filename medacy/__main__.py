import argparse
import logging
from datetime import datetime
import time

from medacy.data import Dataset
from medacy.ner import Model
from medacy.ner import SpacyModel
from medacy.ner.pipelines import LstmClinicalPipeline
from medacy.ner.pipelines import ClinicalPipeline

def setup(args):
    dataset = Dataset(args.dataset)

    pipeline = None
    if args.pipeline == 'bilstm-clinical':
        labels = list(dataset.get_labels())
        pipeline = LstmClinicalPipeline(entities=labels)
    elif args.pipeline == 'spacy':
        model = SpacyModel()
        return dataset, model
    elif args.pipeline == 'clinical':
        labels = list(dataset.get_labels())
        pipeline = ClinicalPipeline(entities=labels)
    else:
        raise TypeError('%s is not a supported pipeline.' % args.pipeline)

    model = Model(pipeline)

    return dataset, model

def train(args, dataset, model):
    model.fit(dataset)
    model.dump(args.filename)

def predict(args, dataset, model):
    model.load(args.model_path)
    model.predict(dataset)

def cross_validate(args, dataset, model):
    model.cross_validate(num_folds=5, training_dataset=dataset)

def main():
    # Argparse setup
    parser = argparse.ArgumentParser(prog='medacy', description='Train and evaluate medaCy pipelines.')
    parser.add_argument('-p', '--print_logs', action='store_true', help='Use to print logs to console.')
    parser.add_argument('-pl', '--pipeline', choices=['bilstm-clinical','clinical', 'spacy'], default='bilstmcrf', help='Pipeline to use for training.')
    parser.add_argument('-d', '--dataset', required=True, help='Directory of dataset to use for training.')
    subparsers = parser.add_subparsers()

    # Train arguments
    parser_train = subparsers.add_parser('train', help='Train a new model.')
    parser_train.add_argument('-f', '--filename', required=True, help='Filename to use for saved model.')
    parser_train.set_defaults(func=train)

    # Predict arguments
    parser_predict = subparsers.add_parser('predict', help='Run predictions on the dataset using a trained model.')
    parser_predict.add_argument('-m', '--model_path', required=True, help='Trained model to load.')
    parser_predict.set_defaults(func=predict)

    # Cross Validation arguments
    parser_validate = subparsers.add_parser('validate', help='Cross validate a model on a given dataset.')
    parser_validate.set_defaults(func=cross_validate)

    # Parse initial args
    args = parser.parse_args()

    # Logging
    logging.basicConfig(filename='medacy.log', level=logging.INFO)
    if args.print_logs:
        logging.getLogger().addHandler(logging.StreamHandler())
    start_time = time.time()
    current_time = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('\nSTART TIME: ' + current_time)

    # Run proper function
    dataset, model = setup(args)
    args.func(args, dataset, model)

    # Calculate/print end time
    end_time = time.time()
    current_time = datetime.fromtimestamp(end_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('END TIME: ' + current_time)

    # Calculate/print time elapsed
    seconds_elapsed = end_time - start_time
    minutes_elapsed = seconds_elapsed / 60.0
    hours_elapsed = minutes_elapsed / 60.0
    logging.info('HOURS ELAPSED: %.0f:%.02f' % (hours_elapsed, minutes_elapsed))

if __name__ == '__main__':
    main()
