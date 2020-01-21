
from medacy.data import Dataset
from medacy.pipelines import LstmSystematicReviewPipeline
from medacy.model import Model
import logging
from medacy.data.annotations import Annotations
import os
from shutil import copyfile


def setup(predicting, training, word_embeddings, cuda_device):
    predicting_dataset = Dataset(predicting)
    training_dataset = Dataset(training)

    entities = training_dataset.get_labels()

    pipe = LstmSystematicReviewPipeline(
        entities=entities,
        cuda_device=cuda_device,
        word_embeddings=word_embeddings
    )

    model = Model(pipe)
    return model, pipe, predicting_dataset, training_dataset


def main():
    model, pipe, predicting_dataset, training_dataset = setup(
        predicting='./2010_data',
        training='./merged_n2c2',
        cuda_device=3,
        word_embeddings='../w2v.model'
    )

    model.load('./lstm_n2c2_model.pt')

    # creates the training x_data, which contains the correct number of each other feature (i.e., suffix, prefix...)
    model.preprocess(training_dataset)
    train_x_data = [x[0] for x in model.X_data]

    data_files = [data_file.file_name for data_file in predicting_dataset]
    file_paths = [d.txt_path for d in predicting_dataset]

    # grabbed this code from the preprocess function, but need to keep the docs intact so not using function directly
    predict_docs = [model._run_through_pipeline(data_file, predicting=True) for data_file in predicting_dataset]

    # loop through all the predict_docs and predict the annotations
    for num, doc in enumerate(predict_docs):
        # code from predict_document in _model.py
        feature_extractor = pipe.get_feature_extractor()

        features, labels = model._extract_features(doc)

        # grab features dictionaries and vectorize
        x_data = [x[0] for x in features]
        x_data = model.model.vectorizer.vectorize_prediction_dataset(
            predicting_tokens=x_data, training_tokens=train_x_data)

        _, indices = feature_extractor.get_features_with_span_indices(doc)
        predictions = []
        for sentence in x_data:
            emissions = model.model.model(sentence).unsqueeze(1)
            tag_indices = model.model.model.crf.decode(emissions)
            predictions.append(model.model.vectorizer.devectorize_tag(tag_indices[0]))

        # The rest of the code is basically copy-pasted from _model.py, with minor alterations
        predictions = [element for sentence in predictions for element in sentence]  # flatten 2d list
        span_indices = [element for sentence in indices for element in sentence]  # parallel array containing indices

        annotations = []
        i = 0
        while i < len(predictions):
            if predictions[i] == "O":
                i += 1
                continue
            entity = predictions[i]
            first_start, first_end = span_indices[i]
            # Ensure that consecutive tokens with the same label are merged
            while i < len(predictions) - 1 and predictions[i + 1] == entity:  # If inside entity, keep incrementing
                i += 1
            last_start, last_end = span_indices[i]

            labeled_text = doc.text[first_start:last_end]

            logging.debug("%s: Predicted %s at (%i, %i) %s", doc._.file_name, entity, first_start, last_end,
                          labeled_text.replace('\n', ''))

            annotations.append((entity, first_start, last_end, labeled_text))
            i += 1

        annotations = Annotations(annotations)
        ann_location = os.path.join('../gpu_test', data_files[num] + ".ann")
        annotations.to_ann(write_location=ann_location)

        txt_location = os.path.join('../gpu_test', data_files[num] + ".txt")
        copyfile(file_paths[num], txt_location)


if __name__ == '__main__':
    main()
