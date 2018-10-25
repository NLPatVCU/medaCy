from sklearn_crfsuite import CRF
from .crf_predictor import model_to_ann
import logging, os
"""
Predicts NER output utilizing an existing or pre-built model.
"""
class Predictor():

    def __init__(self, medacy_pipeline, data_loader, model=None):
        self.medacy_pipeline = medacy_pipeline
        self.data_loader = data_loader
        assert isinstance(model, CRF), "MedaCy currently only supports CRF models - model was not an instance of CRF from sklearn_crfsuite."
        self.model = model
        self.prediction_directory = data_loader.data_directory + "/predictions/"

        if os.path.isdir(self.prediction_directory):
            logging.warning("Overwritting existing predictions")
        else:
            os.makedirs(self.prediction_directory)


    def predict(self, model=None):
        if model is not None:
            assert isinstance(model, CRF), "MedaCy currently only supports CRF models - model was not an instance of CRF from sklearn_crfsuite."
            self.model = model
        else:
            model = self.model

        data_loader = self.data_loader
        medacy_pipeline = self.medacy_pipeline

        #create directory to write predictions to.


        for data_file in data_loader.get_files():
            logging.info("Predicting file: %s", data_file.file_name)

            with open(data_file.raw_path, 'r') as raw_text:
                doc = medacy_pipeline.spacy_pipeline.make_doc(raw_text.read())

            if data_file.metamapped_path is not None:
                doc.set_extension('metamapped_file', default=data_file.metamapped_path, force=True)

            #run through the pipeline
            doc = self.medacy_pipeline(doc)

            ann_file_contents = model_to_ann(model, medacy_pipeline, doc)
            with open(self.prediction_directory+data_file.file_name+".ann", "w") as f:
                f.write(ann_file_contents)








