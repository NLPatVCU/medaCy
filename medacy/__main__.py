"""
MedaCy CLI Setup
"""
import argparse
import datetime as dt
import importlib
import json
import logging

from medacy.data.dataset import Dataset
from medacy.model.model import Model, DEFAULT_NUM_FOLDS
from medacy.pipelines import bert_pipeline
from medacy.tools.json_to_pipeline import json_to_pipeline


def setup(args):
    """
    Sets up dataset and pipeline/model since it gets used by every command.
    :param args: Argparse args object.
    :return dataset, model: The dataset and model objects created.
    """
    dataset = Dataset(args.dataset)
    entities = list(dataset.get_labels())
    if args.test_mode:
        dataset.data_limit = 1

    if args.entities is not None:
        with open(args.entities, 'rb') as f:
            data = json.load(f)
        json_entities = data['entities']
        if not set(json_entities) <= set(entities):
            raise ValueError(f"The following entities from the json file are not in the provided dataset: {set(json_entities) - set(entities)}")
        entities = json_entities

    if args.custom_pipeline is not None:
        logging.info(f"Using custom pipeline configured at {args.custom_pipeline}")
        # Construct a pipeline class (not an instance) based on the provided json path;
        # args.custom_pipeline is that path
        Pipeline = json_to_pipeline(args.custom_pipeline)
    else:
        # Parse the argument as a class name in module medacy.pipelines
        module = importlib.import_module("medacy.pipelines")
        Pipeline = getattr(module, args.pipeline)
        logging.info('Using %s', args.pipeline)

    pipeline = Pipeline(
        entities=entities,
        cuda_device=args.cuda,
        word_embeddings=args.word_embeddings,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        pretrained_model=args.pretrained_model,
        using_crf=args.using_crf
    )

    model = Model(pipeline)
    return dataset, model


def train(args, dataset, model):
    """
    Used for training new models.
    :param args: Argparse args object.
    :param dataset: Dataset to use for training.
    :param model: Untrained model object to use.
    """
    if args.filename is None:
        raise RuntimeError("A file name must me specified with -f when training a model")

    model.fit(dataset, groundtruth_directory=args.groundtruth)
    model.dump(args.filename)


def predict(args, dataset, model):
    """
    Used for running predictions on new datasets.
    :param args: Argparse args object.
    :param dataset: Dataset to run prediction over.
    :param model: Trained model to use for predictions.
    """
    if not args.predictions:
        args.predictions = None

    model.load(args.model_path)
    model.predict(
        dataset,
        prediction_directory=args.predictions
    )


def cross_validate(args, dataset, model):
    """
    Used for running k-fold cross validations.
    :param args: Argparse args object.
    :param dataset: Dataset to use for training.
    :param model: Untrained model object to use.
    """
    model.cross_validate(
        num_folds=args.k_folds,
        training_dataset=dataset,
        prediction_directory=args.predictions,
        groundtruth_directory=args.groundtruth
    )


def main():
    """
    Main function where initial argument parsing happens.
    """
    # Argparse setup
    parser = argparse.ArgumentParser(prog='medacy', description='Train, evaluate, and predict with medaCy.')
    # Global variables
    parser.add_argument('-pl', '--pipeline', default='ClinicalPipeline', help='Pipeline to use for training. Write the exact name of the class.')
    parser.add_argument('-cpl', '--custom_pipeline', default=None, help='Path to a json file of a custom pipeline, as an alternative to a medaCy pipeline')
    parser.add_argument('-d', '--dataset', required=True, help='Directory of dataset to use for training.')
    parser.add_argument('-ent', '--entities', default=None,
                        help='Path to a json file containing an \"entities\" key of a list of entities to use; otherwise all the entities in the dataset will be used.')

    # Logging, testing variables
    test_group = parser.add_argument_group('Logging and testing arguments')
    test_group.add_argument('-lc', '--log_console', action='store_true', help='Use to print logs to console.')
    test_group.add_argument('-lf', '--log_file', default=None, help='Specify a log file path, if something other than the default is desired')
    test_group.add_argument('-t', '--test_mode', default=False, action='store_true',
                            help='Specify that the action is a test (automatically uses only a single '
                                 'data file from the dataset and sets logging to debug mode)')

    # GPU-specific
    gpu_group = parser.add_argument_group('GPU Arguments', 'Arguments that relate to the GPU, used by the BiLSTM and BERT')
    gpu_group.add_argument('-c', '--cuda', type=int, default=-1, help='Cuda device to use. -1 to use CPU.')

    # BiLSTM-specific
    bilstm_group = parser.add_argument_group('BiLSTM Arguments', 'Arguments for the BiLSTM learner')
    bilstm_group.add_argument('-w', '--word_embeddings', help='Path to word embeddings, needed for BiLSTM.')

    # BERT-specific
    bert_group = parser.add_argument_group('BERT Arguments', 'Arguments for the BERT learner')
    bert_group.add_argument('-b', '--batch_size', type=int, default=bert_pipeline.BATCH_SIZE, help='Batch size.')
    bert_group.add_argument('-lr', '--learning_rate', type=float, default=bert_pipeline.LEARNING_RATE, help='Learning rate for train and cross validate.')
    bert_group.add_argument('-e', '--epochs', type=int, default=bert_pipeline.EPOCHS, help='Number of epochs to train for.')
    bert_group.add_argument('-pm', '--pretrained_model', type=str, default='bert-large-cased', help='Which pretrained model to use.')
    bert_group.add_argument('-crf', '--using_crf', action='store_true', help='Use a CRF layer.')

    subparsers = parser.add_subparsers()

    # Cross Validation arguments
    parser_validate = subparsers.add_parser('validate', help='Cross validate a model on a given dataset.')
    parser_validate.add_argument('-k', '--k_folds', default=DEFAULT_NUM_FOLDS, type=int, help='Number of folds to use for cross-validation.')
    parser_validate.add_argument('-gt', '--groundtruth', type=str, default=None, help='Directory to write groundtruth files.')
    parser_validate.add_argument('-pd', '--predictions', type=str, default=None, help='Directory to write prediction files.')
    parser_validate.set_defaults(func=cross_validate)

    # Train arguments
    parser_train = subparsers.add_parser('train', help='Train a new model.')
    parser_train.add_argument('-f', '--filename', help='Filename to use for saved model.')
    parser_train.add_argument('-gt', '--groundtruth', type=str, default=None, help='Directory to write groundtruth files.')
    parser_train.set_defaults(func=train)

    # Predict arguments
    parser_predict = subparsers.add_parser('predict', help='Run predictions on the dataset using a trained model.')
    parser_predict.add_argument('-m', '--model_path', required=True, help='Trained model to load.')
    parser_predict.add_argument('-pd', '--predictions', default=None, help='Directory to store prediction files.')
    parser_predict.set_defaults(func=predict)

    # Parse initial args
    args = parser.parse_args()

    # Logging
    device = str(args.cuda) if args.cuda >= 0 else 'cpu'
    log_file_name = args.log_file or f"medacy_{device}.log"
    logging.basicConfig(filename=log_file_name, format='%(asctime)-15s: %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    if args.log_console or args.test_mode:
        logger.addHandler(logging.StreamHandler())
    if args.test_mode:
        logger.setLevel(logging.DEBUG)
        logging.info("Test mode enabled: logging set to debug")
    start_time = dt.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'\n\nSTART TIME: {start_timestamp}')

    # Run proper function
    dataset, model = setup(args)
    args.func(args, dataset, model)

    # Calculate/print end time
    end_time = dt.datetime.now()
    end_timestamp = end_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'END TIME: {end_timestamp}')

    # Calculate/print time elapsed
    elapsed_time: dt.timedelta = end_time - start_time
    logging.info(f'TIME ELAPSED: {elapsed_time}')


if __name__ == '__main__':
    main()
