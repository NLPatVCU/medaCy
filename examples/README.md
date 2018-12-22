# Using medaCy: Tutorials and Workflows
This directory contains common workflows for using medaCy

## Table of contents
1. [How medaCy Works](#how-medacy-works)
2. [Building a medaCy Pipeline](#building-a-custom-medacy-pipeline)
3. [Pre-trained Models](#utilizing-pre-trained-ner-models)

### How medaCy Works
MedaCy leverages the text-processing power of spaCy with state-of-the-art research tools and techniques in medical named entity recognition.

MedaCy consists of a set of lightning-fast pipelines that are specialized for learning specific types of medical entities and relations. A pipeline consists
of a stackable and interchangeable set of PipelineComponents - these are bite-sized code blocks that each overlay a feature onto the text being processed.

#### Pipeline Components
PipelineComponents can be developed to utilize in custom Pipelines by interfacing the [BaseComponent](medacy/pipeline_components/base/base_component.py) and [BasePipeline](medacy/pipelines/base/base_pipeline.py) classes respectively. Alternatively use components already implemented in medaCy. Some more powerful components require outside software - an example is the MetaMapComponent which interfaces with [MetaMap](https://metamap.nlm.nih.gov/)
to overlay rich medical concept information onto text. Components are chained or stacked in pipelines and can themselves depend on the outputs of previous components to function. In the underlying implementation, a medaCy PipelineComponent is a wrapper over a spaCy component that includes a number of utilities specific to faciliting the training, utilization, and distribution process of medical domain text processing models.

### Building a custom medaCy pipeline

### Utilizing Pre-trained NER models
| Application | Dataset Trained Over | Entities |
| :---------: | :----------------: |:-------------:|
| [Clinical Notes](/examples/models/clinical_notes_model.md)| [N2C2 2018](https://n2c2.dbmi.hms.harvard.edu/) | Drug, Form, Route, ADE, Reason, Frequency, Duration, Dosage, Strength  |
| [EPA Systematic Reviews](/examples/models/epa_systematic_review_model.md) | [TAC SRIE 2018](https://tac.nist.gov/2018/SRIE/) | Species, Celline, Dosage, Group, etc. |
| [Nanomedicine Drug Labels](/examples/models/nanomedicine_drug_labels.md) | [END](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5644562/) | Nanoparticle, Company, Adverse Reaction, Active Ingredient, Surface Coating, etc. |
