from __future__ import unicode_literals, print_function

from os import listdir
from os.path import isfile, join
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from medacy.tools import Annotations, DataFile

# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    input_dir=("Directory of ann and txt files.", "option", "i", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="animal", input_dir=None, output_dir=None, n_iter=30):
    """ Updates a spaCy with additional ner training. Currently only using a set of brat files as
        the training data.
    """
    labels = set()
    train_data = []
    dataset_directory = input_dir
    names = set()

    # Get list of shared names for each set of .ann and .txt file
    for filename in listdir(dataset_directory):
        path = join(dataset_directory, filename)

        if isfile(path):
            names.add(filename.split('.')[0])

    # add error checking for missing .ann or .txt files

    # Add each entry in dataset with annotation to train_data
    for name in names:
        txt_path = join(dataset_directory, name + '.txt')
        ann_path = join(dataset_directory, name + '.ann')
        annotations = Annotations(ann_path, source_text_path=txt_path)
        train_data.append(annotations.get_entity_annotations(format='spacy'))
        labels.update(annotations.get_labels())

    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")
        print(ner.labels)

    for label in labels:
        ner.add_label(label)

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "I prescribed them 128mg of adderall"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
