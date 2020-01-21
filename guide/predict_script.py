
from medacy.data import Dataset
from medacy.pipelines import LstmSystematicReviewPipeline
from medacy.model import Model
import logging
from medacy.data.annotations import Annotations
import os
from shutil import copyfile


def main():
    # build medacy dataset for the data the model was trained on, and the data being predicted over
    trained_dataset = Dataset('./data')
    predict_dataset = Dataset('./data_text')

    # used later for copying .txt files over to prediction directory
    file_paths = [d.txt_path for d in predict_dataset]

    # hardcoded n2c2 entities
    entities = ["Strength", "ADE", "Duration", "Dosage", "Frequency", "Reason", "Form", "Drug", "Route"]

    pipeline = LstmSystematicReviewPipeline(
        entities=entities,
        word_embeddings='../medinify/data/embeddings/w2v.model',
        cuda_device=-1
    )

    # load trained model
    model = Model(pipeline)
    model.load('./data_model.pt')
    # when we vectorize the data later, this variable will be changed, and need to be reset
    tag_to_index = model.model.vectorizer.tag_to_index

    # creates the training x_data, which contains the correct number of each other feature (i.e., suffix, prefix...)
    model.preprocess(trained_dataset)

    predict_docs = [model._run_through_pipeline(data_file, predicting=True) for data_file in predict_dataset]
    train_x_data = [x[0] for x in model.X_data]
    train_y_data = model.y_data

    data_files = [data_file.file_name for data_file in predict_dataset]

    # loop through all the predict_docs and predict the annotations
    for num, doc in enumerate(predict_docs):
        feature_extractor = pipeline.get_feature_extractor()
        _, indices = feature_extractor.get_features_with_span_indices(doc)

        features, labels = model._extract_features(doc)

        x_data = [x[0] for x in features]
        x_data = model.model.vectorizer.vectorize_prediction_tokens(
            predicting_tokens=x_data, training_tokens=train_x_data, train_y_data=train_y_data)

        model.model.vectorizer.tag_to_index = tag_to_index

        predictions = []
        for sentence in x_data:
            emissions = model.model.model(sentence).unsqueeze(1)
            tag_indices = model.model.model.crf.decode(emissions)
            predictions.append(model.model.vectorizer.devectorize_tag(tag_indices[0]))
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
        ann_location = os.path.join('./practice_predictions', data_files[num] + ".ann")
        annotations.to_ann(write_location=ann_location)

        txt_location = os.path.join('./practice_predictions', data_files[num] + ".txt")
        copyfile(file_paths[num], txt_location)


if __name__ == '__main__':
    main()
