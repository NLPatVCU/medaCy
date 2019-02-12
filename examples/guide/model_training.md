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
from medacy.data import Dataset
from medacy.pipelines import ClinicalPipeline
from medacy.model import Model

entities = ['Drug', 'Strength']

training_dataset = Dataset('/home/medacy/clinical_training_data/')
pipeline = ClinicalPipeline(metamap=None, entities=entities)
model = Model(pipeline, n_jobs=30) #distribute documents between 30 processes during training and prediction

model.fit(training_dataset)

model.dump('/home/medacy/clinical_model.pickle')


```

The `ClinicalPipeline` source looks like this:

```python
import spacy, sklearn_crfsuite
from .base import BasePipeline
from ..pipeline_components import ClinicalTokenizer
from medacy.model.feature_extractor import FeatureExtractor

from ..pipeline_components import GoldAnnotatorComponent, MetaMapComponent, UnitComponent, MetaMap

class ClinicalPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. A special tokenizer that breaks down a clinical document
    to character level tokens defines this pipeline.
    """

    def __init__(self, metamap=None, entities=[]):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap if metamap should be used, defaults to None.
        """
        description="""Pipeline tuned for the extraction of ADE related entities from the 2018 N2C2 Shared Task"""
        super().__init__("clinical_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=description,
                         creators="Andriy Mulyar (andriymulyar.com)", #append if multiple creators
                         organization="NLP@VCU"
                         )

        self.entities = entities

        self.spacy_pipeline.tokenizer = self.get_tokenizer() #set tokenizer

        self.add_component(GoldAnnotatorComponent, entities) #add overlay for GoldAnnotation

        if metamap is not None and isinstance(metamap, MetaMap):
            self.add_component(MetaMapComponent, metamap)

        #self.add_component(UnitComponent)


    def get_learner(self):
        return ("CRF_l2sgd", sklearn_crfsuite.CRF(
            algorithm='l2sgd',
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        ))

    def get_tokenizer(self):
        tokenizer = ClinicalTokenizer(self.spacy_pipeline)
        return tokenizer.tokenizer

    def get_feature_extractor(self):
        extractor = FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
        return extractor
```


The `__init__` method defines pipeline meta-data along with initializing the sequence of components the pipeline will use to annotate custom token attributes over the document. Components are imported and initialized as part of the pipeline by calling the `add_component` method. The first paramater is a component and the subsequent parameters are any arguments that are passed to the component on initialization. Token attributes beginning with `feature_` are automically collected by the `FeatureExtractor` initialized in the `get_feature_extractor` method.  Note the instantiation of the `FeatureExtractor` allows the definition of an array of `spacy_features` to utilize - these can be any attribute of a spaCy [Token](https://spacy.io/api/token#attributes).

The `get_learner` method returns a configured instance of the machine learning algorithm to utilize for training a model. Currently only CRF models wrapped by the package [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/) are allowed.

The `get_tokenizer` method returns a configured medaCy tokenizer. An interface for building and maintaining a tokenizer is provided and the pattern from `ClinicalTokenizer` can be followed for engineering your own.

The `get_feature_extractor` method returns a configured feature extractor. This defines how and what features from annotated documents are collected to be fed into the model during training or prediction. The example configuration means that all medaCy annotated features and the specified `spacy_features` are collected in a range of three tokens to the left and three tokens to the right of every token (ie. the `window_size`).





