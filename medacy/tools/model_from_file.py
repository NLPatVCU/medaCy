import logging, os, joblib
from medacy.pipelines.base.base_pipeline import BasePipeline
from medacy.models.base.base_model import BaseModel
"""
Takes a filepath containing a pickled representation of a Model object containing its pipeline, model, and model description.
Returns an initialized model object containing the specifications read in from the pickled object.
"""

# TODO Finish implementing loading in and initializing a Model object from a stored description
def model_frome_file(filepath):
    assert os.path.exists(filepath), "Cannot find file or directory " + filepath
    logging.info("Initializing model from file at %s", filepath)
    with open(filepath, 'rb') as f:
        model_description = joblib.load(f)
        pipeline = model_description['pipeline']
        model = model_description['model']
        model_name = model.description['model_name']
        assert isinstance(pipeline, BasePipeline), "Valid MedaCy Model objects contain a MedaCy Pipeline"
