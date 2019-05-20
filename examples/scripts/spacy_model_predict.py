"""Example script for running a prediction with spaCy model.
"""
from pathlib import Path
import plac
from medacy.data import Dataset
from medacy.ner import SpacyModel

# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

@plac.annotations(
    spacy_model_path=("Path to spaCy model to load", "option", "m", Path),
    input_dir=("Directory of ann and txt files to predict for", "option", "i", Path),
)
def main(spacy_model_path, input_dir):
    """Main function.
    """
    dataset = Dataset(input_dir)
    model = SpacyModel()
    model.load(spacy_model_path)
    model.predict(dataset)

if __name__ == "__main__":
    plac.call(main)
