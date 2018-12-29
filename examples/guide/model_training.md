# Training a medaCy Model
The power of medaCy resides in its model training ability. While closely following spaCy's [Language](https://spacy.io/usage/adding-languages) architecture, medaCy allows the inclusion of features, machine learning algorithms, functionalities not supported by spaCy. Every medaCy pipeline is built overtop of a spaCy language and can thus interface and utilize any components of that Language (such as lemmatizers, POS tagges and dependency parsers).

Any model trained with medaCy can be [saved, versioned, and easily distributed](packaging_a_medacy_model.md).

In training a medaCy model you are concerned with three things:

1. [Text tokenization](#tokenization)
2. [Feature Overlaying and Token Merging](#feature-overlaying-and-token-merging)
3. [Feature Extraction](#feature-extraction)
4. [Model Selection and Tuning](#model-selection)

These components co-exist in a *Pipeline* - a medaCy model is hence simply a fine-tuned configuration of these components for the task at hand.


## Tokenization
When a document enters medaCy, it is first tokenized. MedaCy does not work with the original document text, but rather an atomized version of the document. Atomized tokens are what the underlying machine learning algorithm is trained to classify - that is, a proper tokenization is **vital** to a highly predictive model.

Proper document tokenization depends on the domain. In some domains text naturally lends to being tokenized by spaces (think plain english) while other domains require a bit more sophistication. For instance, medaCy's [Clinical Model](../models/clinical_notes_model.md) utilizes a character level tokenization. This means that the end learning algorithm classifies sequences of characters - not words.

medaCy utilizes the tokenization functionality provided by spaCy. That is, a set of tokenization rules is defined and then utilized to transform the document text into a spaCy [Doc](https://spacy.io/api/doc) object. In short, a [Doc](https://spacy.io/api/doc) object wraps C code that allows for its ightning fast manipulation.

Select a medaCy pre-configured tokenizer or [build your own](building_a_custom_tokenizer.md).

## Feature Overlaying and Token Merging
After a document has been tokenized, it is run through a sequence of *pipeline_components*. A *pipeline_component* overlays features or attributes onto tokens in the form of spaCy [custom token extensions](https://spacy.io/api/token#set_extension). Models trained utilizing spaCy's [thinc](https://github.com/explosion/thinc) package **do not** allow for the use of custom token attributes as features - medaCy adds this functionality. Learn [how to build a medaCy pipeline component](building_a_custom_pipeline_component.md). When appending a custom attribute to be used as a feature, simply prefix the attribute with `feature_` and it will automatically be detected and utilized during model training.

Alongside overlaying features, pipeline components can be utilized for the merging of tokens. Inside a pipeline component, a spaCy [Matcher](https://spacy.io/api/matcher) can be utilized to group tokens together based on patterns. This greatly decreases the amount of extraneous information passed into the end learning algorithm effectively decreasing noise and increasing predictive performance. Each grouped pattern can assigned a new custom attribute that can then also be used as a feature during training. Token merging based on outside lexicons alongside feature annotating is an effective way to increase the *precision* of your trained models while still allowing for the incorporation of information not included in the lexicons.



## Feature Extraction



