import logging
"""
Takes a model and a document to annotate
Outputs a string containing a proper.ann file with the model tags
"""

def model_to_ann(model, medacy_pipeline, doc):
    """

    :param crf_model: a built model
    :param doc: a document that has already ran through the pipeline
    :return: a string containing a valid .ann file
    """

    extract = medacy_pipeline.get_feature_extractor()

    features, indices = extract.get_features_with_span_indices(doc)


    predictions = model.predict(features)

    predictions = [element for sentence in predictions for element in sentence] #flatten 2d list

    span_indices = [element for sentence in indices for element in sentence]

    ann_file = ""
    num = 1

    i=0

    while i < len(predictions):
        if predictions[i] == "O":
            i+=1
            continue
        entity = predictions[i]

        #insure that consecutive tokens with the same label are merged

        first_start, first_end = span_indices[i]

        while i < len(predictions)-1 and predictions[i+1] == entity:
            i+=1

        last_start, last_end = span_indices[i]


        #print(entity,":",  doc.text[first_start:last_end], first_start, last_end)

        labeled_text = doc.text[first_start:last_end]

        logging.info("Writing prediction: %s %s", entity, labeled_text.replace('\n', ' '))

        ann_file += "T%i\t%s %i %i\t%s\n" % (num, entity, first_start, last_end, labeled_text.replace('\n', ' '))
        num+=1
        i+=1


    return ann_file
