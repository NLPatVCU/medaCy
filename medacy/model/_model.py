"""
Utilities for methods of the Model class
"""
import logging
from medacy.data.annotations import Annotations


def predict_document(model, doc, medacy_pipeline):
    """
    Generates an dictionary of predictions of the given model over the corresponding document. The passed document
    is assumed to be annotated by the same pipeline utilized when training the model.
    :param model: A loaded medaCy NER model
    :param doc: A spacy document
    :param medacy_pipeline: An instance of a medacy pipeline
    :return: an Annotations object containing the model predictions
    """
    #assert isinstance(feature_extractor, FeatureExtractor), "feature_extractor must be an instance of FeatureExtractor"

    feature_extractor = medacy_pipeline.get_feature_extractor()

    features, indices = feature_extractor.get_features_with_span_indices(doc)
    predictions = model.predict(features)
    predictions = [element for sentence in predictions for element in sentence]  # flatten 2d list
    span_indices = [element for sentence in indices for element in sentence] #parallel array containing indices
    annotations = []

    i = 0
    while i < len(predictions):
        if predictions[i] == "O":
            i += 1
            continue
        entity = predictions[i]
        first_start, first_end = span_indices[i]
        # Ensure that consecutive tokens with the same label are merged
        while i < len(predictions)-1 and predictions[i+1] == entity: #If inside entity, keep incrementing
            i += 1
        last_start, last_end = span_indices[i]

        labeled_text = doc.text[first_start:last_end]

        logging.debug("%s: Predicted %s at (%i, %i) %s", doc._.file_name, entity, first_start, last_end ,labeled_text.replace('\n', ''))

        annotations.append((entity, first_start, last_end, labeled_text))
        i += 1

    return Annotations(annotations)


def construct_annotations_from_tuples(doc, predictions):
    """
    Converts predictions mapped to a document into an Annotations object
    :param doc: SpaCy doc corresponding to predictions
    :param predictions: List of tuples containing (entity, start offset, end offset)
    :return: Annotations Object representing predicted entities for the given doc
    """
    predictions = sorted(predictions, key=lambda x: x[1])
    annotations = []

    for prediction in predictions:
        if len(prediction) == 3:
            (entity, start, end) = prediction
            labeled_text = doc.text[start:end]
        elif len(prediction) == 4:
            (entity, start, end, labeled_text) = prediction
        else:
            raise ValueError("Incorrect prediction length.")

        annotations.append((entity, start, end, labeled_text))

    return Annotations(annotations)
