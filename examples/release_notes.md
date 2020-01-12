# MedaCy Release Notes

- The package has been restructured. 
Reference the following file tree to locate classes you may have been using.

```
├── data
    ├── Annotations
    ├── DataFile
    └── Dataset
├── model
    ├── Model
    └── SpacyModel
├── pipeline_components
    ├── feature_extractors
    ├── feature_overlayers
        └── MetaMap
    ├── learners
        ├── BiLSTM+CRF
        └── BERT
    └── tokenizers
├── pipelines
└── tools
    ├── calculators
    ├── converters
    └── json_to_pipeline 
```

- Annotations
    - Annotations objects only represent Entity annotations and wrap around a list of entity tuples
    rather than a dictionary of entities and relations.
    - Two Annotations objects can be merged with the pipe operator.
    - Entities with non-contiguous spans can now be represented, but the space between the spans 
    is also included.
- Dataset
    - `dataset.generate_annotations()` has been created, allowing for iterative accessing of 
    Annotations objects for each entry in the Dataset.
    - Dict-like lookup of Annotations objects is supported with `dataset['some_file']`
    returning an Annotations object of `some_file.ann` if it exists in the directory for 
    the given Dataset, allowing for easy access across parallel Dataset instances.
- MultiModel
    - This new class allows for prediction with multiple models over directories in a 
    memory-efficient way.
- MetaMap
    - Models created using MetaMap-overlayed features can now be used for prediction, 
    provided that MetaMap is running.
    - MetaMap can be activated or deactivated within a Python script using `activate` and
    `deactivate` methods of MetaMap instances, or by using a `with` statement.
- json_to_pipeline
    - This function allows for custom pipeline configuration using a JSON file and medaCy's
    command line interface.