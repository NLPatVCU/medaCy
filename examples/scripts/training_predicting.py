# This script demonstrates utilizing medaCy for a full model training/predictive/cross validation use-case.
# > python training_predicting.py model_name
# Will build a model named model_name with the pipeline and parameters defined below. This script places the model in
# it's own directory along the models build log and model/pipeline parameters to keep results easily referencable during run time.
# Once a sufficent model is produced, consider wrapping it up into a medaCy compatible model as defined the example guide.

import datetime
import logging
import os
import sys
import time

from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipelines.systematic_review_pipeline import SystematicReviewPipeline

train_dataset, evaluation_dataset = Dataset.load_external('medacy_dataset_tac_2018')
entities = train_dataset.get_labels(as_list=True)

if sys.argv[1] is None:
    exit(0)
    
# For rapid model prototyping, will train and predict by simply running the script with a model name as a parameter.
model_name = sys.argv[1]  # name for the model, use underscores
model_notes = "notes about the current model"  # notes about current model to be stored in a model information file by this script.

model_directory = "/home/username/named_entity_recognition/challenges/challenge_n/models/%s" % model_name.replace(" ", '_')

if model_name == "" or os.path.isdir(model_directory):
    print("Model directory already exists, aborting")
    exit(0)
else:
    os.mkdir(model_directory)

current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H.%M.%S')
logging.basicConfig(filename=model_directory+'/build_%s.log' % current_time, level=logging.DEBUG)


# Initialize everything needed for model

# Metamaps the dataset, if it not already, and stores the metamapped files for access in training_dataset.
# See Dataset API for details.
metamap = MetaMap("/home/share/programs/metamap/2016/public_mm/bin/metamap")
with metamap:
    train_dataset.metamap(metamap, n_jobs=3)
    evaluation_dataset.metamap(metamap, n_jobs=3)

# Selects the pre-processing pipeline this model should be trained with respect to.
pipeline = SystematicReviewPipeline(entities=entities, use_metamap=True)
model = Model(pipeline, n_jobs=1)
# number of cores to utilize during feature extraction when training the model.
# Note: this is done by forking, not threading hence utlizes a large amount of memory.

# Write information about model before training
with open(model_directory+"/model_information.txt", 'w') as model_info:
    model_info.write("Entities: [%s]\n" % ", ".join(entities))
    model_info.write("Training Files: %i\n" % len(train_dataset.get_data_files()))
    model_info.write(model_notes+"\n")
    model_info.write(str(model))

model.fit(train_dataset)

# dump fitted model
current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H.%M.%S')
model.dump(model_directory+"/tac_2018_%s_%s.pkl" % (model_name, current_time))

# predicts over the datasets in evaluation_dataset utilizing the model trained above, then stores those predictions
# in a given output directory
model.predict(evaluation_dataset, prediction_directory=os.path.join(model_directory, 'predictions'))

# performs sequence stratified cross validation over the trained model.
# Note that all extracted features are stored in memory while this runs.
model.cross_validate(training_dataset=train_dataset)
