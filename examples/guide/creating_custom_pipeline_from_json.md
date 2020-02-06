# Creating a Pipeline from a JSON file

MedaCy is highly modular and allows for customization of nearly every aspect of the pipeline, including creating pipelines
with custom feature sets. Creating a new pipeline component requires knowledge of developing in Python. As an alternative
for non-Python developers, medaCy's command line interface allows for custom pipeline creation in JSON format using pipeline components
that already exist in medaCy. These custom pipelines allow you to select the learning algorithm, spaCy features, and more.

## Example JSON

```json
{
    "learner": "CRF",
    "spacy_pipeline": "en_core_web_sm",
    "spacy_features": ["text", "pos_", "shape_"],
    "window_size": 3,
    "tokenizer": "systematic_review",
    "metamap": "path/to/metamap/"
}
```

## Options
These keys are required:
* `"entities"`: a list of entities (see following section for a shortcut)
* `"spacy_pipeline"`: name of a spaCy model, see [here](https://spacy.io/usage/models) for a list of available models
* `"spacy_features`: a list of spaCy features to use, see [here](https://spacy.io/api/token) for a complete list
* `"window_size"`: the number of tokens before and after a given word whose features should be considered along 
with the target word; set to `0` to only consider the target word
* `"learner"`: `"CRF"`, `"BiLSTM"`, or `"BERT"`

These keys are optional, and have default behavior if the key is not present:
* `"tokenizer"`: defaults to using the tokenizer from the selected spaCy pipeline; custom options are `"clinical"`,
`"systematic_review"`, or `"character"`
* `"metamap"`: MetaMap will only be used if this key is present; the value should be the path to the MetaMap binary file

Finally, `"metamap"` requires the key `"semantic_types"`, which can be `"all"` to use all semantic types found in the dataset,
`"none"` to not use them, or a list of semantic types to use.


## Usage
The following command will run five-fold cross validation using a custom JSON pipeline:
```bash
(medacy_venv) $ python -m medacy -d /path/to/your/dataset -cpl /path/to/your/pipeline.json validate
```

The BiLSTM and BERT learners require additional command line arguments. Run `python -m medacy --help` for help with 
the command line interface.