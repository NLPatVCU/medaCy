from nlp.clinical_pipeline.feature_extractor import FeatureExtractor
"""
Takes a CRF model and a document to annotate
Outputs a string containing a proper.ann file with the models tags
"""


#TODO refactor completely. This should take a Pipeline and output the build ann file using that pipeline
def model_to_ann(crf_model, doc):
    """

    :param crf_model: a built model
    :param doc: a document that has already ran through the pipeline
    :return: a string containing a valid .ann file
    """

    extract = FeatureExtractor()

    features, indices = extract.get_features_with_span_indices(doc)

    predictions = crf_model.predict(features)

    predictions = [element for sentence in predictions for element in sentence] #flatten 2d list

    span_indices = [element for sentence in indices for element in sentence]

    ann_file = ""
    num = 1

    i=0
    #print(predictions)
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

        print(entity, labeled_text, doc._.metamapped_file.split('/')[-1] )

        ann_file += "T%i\t%s %i %i\t%s\n" % (num, entity, first_start, last_end, labeled_text.replace('\n', ' '))
        num+=1
        i+=1


    return ann_file




