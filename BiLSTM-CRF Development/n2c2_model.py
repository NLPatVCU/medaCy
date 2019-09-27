from medacy.data import Dataset
from medacy.ner.model import Model
from medacy.ner.pipelines import LstmSystematicReviewPipeline
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

n2c2_dataset = Dataset(data_directory="/Users/annaconte/NLPatVCU/Datasets/N2C2_Data", data_limit=1)

pipeline = LstmSystematicReviewPipeline(entities=['ADE', 'Dosage', 'Drug'],
                                        word_embeddings='/Users/annaconte/NLPatVCU/medaCy/medacy/tests/ner/model/test_word_embeddings.txt')

model = Model(pipeline)

model.cross_validate(training_dataset=n2c2_dataset, num_folds=2)

# If evaluation data is available, validate as follows:

# model.predict(testing, prediction_directory = "")
# prediction_datasets = Dataset("path")
# testing.compute_confustion_matrix(prediction_datasets)


# for file in training:
#     print(file.get_annotation_path)

#pprint(training.compute_counts())


#word_embeddings='/Users/annaconte/NLPatVCU/Datasets/mimic3_d200.bin'






