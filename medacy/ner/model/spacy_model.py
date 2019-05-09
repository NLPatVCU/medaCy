from os import listdir
from os.path import isfile, join
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from medacy.tools import Annotations, DataFile
from medacy.data import Dataset

class SpacyModel:
    def fit(self, dataset, spacy_model_name, new_model_name, output_dir, iterations=30):
        """ Updates a spaCy model with additional ner training. Currently only using a set of brat files as
            the training data.
        """
        data_files = dataset.get_data_files()
        labels = dataset.get_labels()
        train_data = dataset.get_training_data()

        print(labels)

        """Set up the pipeline and entity recognizer, and train the new entity."""
        random.seed(0)
        if spacy_model_name is not None:
            nlp = spacy.load(spacy_model_name)  # load existing spaCy model
            print("Loaded model '%s'" % spacy_model_name)
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

        if spacy_model_name is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()
        move_names = list(ner.move_names)
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):  # only train NER
            sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            for itn in range(iterations):
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