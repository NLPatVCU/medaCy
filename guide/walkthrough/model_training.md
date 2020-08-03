# Training a medaCy Model
The power of medaCy resides in its model training ability. While closely following spaCy's [Language](https://spacy.io/usage/adding-languages) architecture, medaCy allows the inclusion of features, machine learning algorithms, and functionalities not supported by spaCy. Every medaCy pipeline is built overtop of a spaCy Language and can thus interface and utilize any components of that Language (such as lemmatizers, POS taggers, and dependency parsers).

Any model trained with medaCy can be [saved, versioned, and easily distributed](packaging_a_medacy_model.md).

In training a medaCy model you are concerned with the following:

1. [Text tokenization](#tokenization)
2. [Feature Overlaying and Token Merging](#feature-overlaying-and-token-merging)
3. [Feature Extraction](#feature-extraction)
4. [Model Selection and Tuning](#model-selection)
5. [Bringing it all Together](#bringing-it-all-together)

These components co-exist in a *Pipeline* - a medaCy model is hence simply a fine-tuned configuration of these components for the task at hand.


## Tokenization
When a document enters medaCy, it is first tokenized. MedaCy does not work with the original document text, but rather an atomized version of the document. Atomized tokens are what the underlying machine learning algorithm is trained to classify - that is, a proper tokenization is **vital** to a highly predictive model.

Proper document tokenization depends on the domain. In some domains text naturally lends to being tokenized by spaces (think plain English) while other domains require a bit more sophistication. For instance, medaCy's [Clinical Model](../models/clinical_notes_model.md) utilizes a character level tokenization. This means that the end learning algorithm classifies sequences of characters - not words.

medaCy utilizes the tokenization functionality provided by spaCy. That is, a set of tokenization rules is defined and then utilized to transform the document text into a spaCy [Doc](https://spacy.io/api/doc) object. In short, a [Doc](https://spacy.io/api/doc) object wraps C code that allows for its lightning fast manipulation.

Select a medaCy pre-configured tokenizer or [build your own](building_a_custom_tokenizer.md).

## Feature Overlaying and Token Merging
After a document has been tokenized, it is run through a sequence of *pipeline_components*. A *pipeline_component* overlays features or attributes onto tokens in the form of spaCy [custom token extensions](https://spacy.io/api/token#set_extension). Models trained utilizing spaCy's [thinc](https://github.com/explosion/thinc) package **do not** allow for the use of custom token attributes as features - medaCy adds this functionality. Learn [how to build a medaCy pipeline component](building_a_custom_pipeline_component.md). When appending a custom attribute to be used as a feature, simply prefix the attribute with `feature_` and it will automatically be detected and utilized during model training.

Alongside overlaying features, pipeline components can be utilized for the merging of tokens. Inside a pipeline component, a spaCy [Matcher](https://spacy.io/api/matcher) can be utilized to group tokens together based on patterns. This greatly decreases the amount of extraneous information passed into the end learning algorithm effectively decreasing noise and increasing predictive performance. Each grouped pattern can assigned a new custom attribute that can then also be used as a feature during training. Token merging based on outside lexicons alongside feature annotating is an effective way to increase the *precision* of your trained models while still allowing for the incorporation of information not included in the lexicons.



## Feature Extraction

As stated in the [previous section](#feature-overlaying-and-token-merging), a document composed of tokens emerges out of a pipeline with custom attributes set on each token. The medaCy *FeatureExtractor* utilizes these custom attributes as features alongside attributes that are set by spaCy. The *FeatureExtractor* identifies token attributes that are meant to used as features during model training by collecting all custom attributes prefixed with `feature_`. To give tokens context, for each  token the *FeatureExtractor* gathers attributes of neighboring tokens in a `window_size`. Varying the `window_size` parameter of the *FeatureExtractor* varies the contextual cues given to the underlying model: too small of a `window_size` and powerful contextual information is lost; to large, and model crippling noise overshadows meaningful attributes during model induction. Remember to keep in mind the tokenizer used by your pipeline when selecting an appropriate `window_size` - a tokenizer that atomizes your document into characters fed into a FeatureExtractor with a small `window_size` would clearly not be effective at even extracting single word entities let alone multi-word phrases. 

## Model selection
By default, medaCy utilizes a discriminative [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)(CRF's) for classification. CRF's are a class of machine learning algorithms capable of inducing highly predictive models when presented with sequence data rich in interdependency between labels and features. Sequence data is characterized by sequences of objects each with a corresponding label and set of observations. For instance consider the following sentences tokenized by spaces:

```
The patient was given tylenol for her headache. Later that day she experienced nausea.
```

the corresponding representation fed into the CRF would correspond to two token sequences:

characterized by features

```
[[{'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}], [{'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}, {'feature1':value1, 'feature2':value2}]]
```
and parallel labels:

```
[['Other', 'Other', 'Other', 'Other', 'Drug', 'Other, 'Other, 'Reason'], ['Other', 'Other', 'Other', 'Other', 'Other', 'ADE' ]]
```

CRF's discriminatively approximate parameters to a probability distribution over labels with priors given by corresponding features. Come time for prediction, the token label maximizing log-likelihood given its feature representation is selected as the model prediction.

By default, medaCy merges consecutive tokens with equivalent predicted labels into single predicted phrases.


## Bringing it all together
The previously mentioned components make up a medaCy model. In summary training a medaCy model looks like this - this example utilizes the `ClinicalPipeline` included in medaCy *without* `MetaMap` enabled:

```python
import os
from medacy.data.dataset import Dataset
from medacy.pipelines import ClinicalPipeline
from medacy.model.model import Model

entities = ['Drug', 'Strength']

training_dataset = Dataset('/home/medacy/clinical_training_data/')
pipeline = ClinicalPipeline(metamap=None, entities=entities)
model = Model(pipeline)

output_file_path = '/home/medacy/clinical_model.pickle'
# Protect against running fit() without having a valid place to save it
assert os.path.isfile(output_file_path)

model.fit(training_dataset)

model.dump(output_file_path)
```
