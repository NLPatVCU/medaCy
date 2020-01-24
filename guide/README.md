# Using medaCy: Tutorials and Workflows
This directory contains common workflows for using medaCy

## Table of contents
1. [How medaCy Works](#how-medacy-works)
2. [Building a medaCy Pipeline](/guide/walkthrough)
3. [Pre-trained Models](#utilizing-pre-trained-ner-models)
4. [Distributing Trained Models](#sharing-your-medacy-models)
5. [Interaction with spaCy](#how-medacy-uses-spacy)

### How medaCy Works
MedaCy leverages the text-processing power of spaCy with state-of-the-art research tools and techniques in medical text mining.
MedaCy consists of a set of lightning-fast pipelines that are specialized for learning specific types of medical entities and relations. A pipeline consists
of a stackable and interchangeable set of PipelineComponents - these are bite-sized code blocks that each overlay a feature onto the text being processed. 

#### Pipeline Components
PipelineComponents can be developed to utilize in custom Pipelines by interfacing the [BaseOverlayer](medacy/pipeline_components/base/base_component.py) and [BasePipeline](medacy/pipelines/base/base_pipeline.py) classes respectively. Alternatively use components already implemented in medaCy. Some more powerful components require outside software - an example is the MetaMapOverlayer which interfaces with [MetaMap](https://metamap.nlm.nih.gov/)
to overlay rich medical concept information onto text. Components are chained or stacked in pipelines and can themselves depend on the outputs of previous components to function. In the underlying implementation, a medaCy PipelineComponent is a wrapper over a spaCy component that includes a number of utilities specific to faciliting the training, utilization, and distribution process of medical domain text processing models.

### Utilizing Pre-trained NER models
To run a medaCy pre-trained model over your own data, simply install the package associated with the model by following the links below. Models officially supported by medacy all start with the prefix *medacy_model*.
For example, assuming you have medaCy installed:

Run:

`pip install git+https://github.com/NLPatVCU/medaCy_model_clinical_notes.git`

then the code snippet


```python
import medacy_model_clinical_notes
model = medacy_model_clinical_notes.load()
model.predict("The patient was prescribed 1 capsule of Advil for 5 days.")
```

will output:
```python
[
    ('Drug', 40, 45, 'Advil'),
    ('Dosage', 27, 28, '1'), 
    ('Form', 29, 36, 'capsule'), 
    ('Duration', 46, 56, 'for 5 days')
]
```

*NOTE: If you are doing bulk prediction over many files at once, it is advisable to utilize the bulk prediction functionality.*

#### List of medaCy pre-trained models
| Application | Dataset Trained Over | Entities |
| :---------: | :----------------: |:-------------:|
| [Clinical Notes](/guide/models/clinical_notes_model.md)| [N2C2 2018](https://n2c2.dbmi.hms.harvard.edu/) | Drug, Form, Route, ADE, Reason, Frequency, Duration, Dosage, Strength  |
| [EPA Systematic Reviews](/guide/models/epa_systematic_review_model.md) | [TAC SRIE 2018](https://tac.nist.gov/2018/SRIE/) | Species, Celline, Dosage, Group, etc. |
| [Nanomedicine Drug Labels](/guide/models/nanomedicine_drug_labels.md) | [END](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5644562/) | Nanoparticle, Company, Adverse Reaction, Active Ingredient, Surface Coating, etc. |


### Sharing your medaCy models
MedaCy models can be packaged and shared with anyone (or no one!) at ease. See [this example](/guide/walkthrough/model_utilization.md) for details.

### How medaCy uses spaCy
[SpaCy](https://github.com/explosion/spaCy) is an open source python package built with cython that allows for lighting fast text processing. MedaCy combines spaCy's memory efficient text processing architecture with tools, ideas and principles from both machine learning and medical computational linguistics to provide a unified framework for researchers and practioners alike to advance medical text mining.
