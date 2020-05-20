# Using our medaCy Trained Model

Once a medaCy model has been trained - it can be saved, distributed, and used for prediction.
To predict with a trained learning model, it is required that documents be run through a pipeline identical to that which the model was trained with. Fortunately, medaCy encapsulates this functionality via the `Model` class.


## Loading a trained model for prediction
Once a CRF model has been trained and saved to disk, it can be loaded again for use by configuring a `Model` object with the pipeline used to train the CRF model. The below example shows how to configure the medaCy clinical model.

```python
from medacy.pipelines import ClinicalPipeline
from medacy.model.model import Model

pipeline = ClinicalPipeline(metamap=None, entities=['Drug'])
model = Model(pipeline)
model.load('/home/medacy/trained_model.pickle')

annotation = model.predict("The patient took 5 mg of aspirin.")  # Returns an Annotations object
```

Model prediction over a string returns a medaCy `Annotations` object. 
Useful functionalities are provided in the `Annotation` class such as the ability to see a *diff* between 
two annotations for empirical analysis.

Trained CRF models are [pickled](https://docs.python.org/3/library/pickle.html) (serialized) binary files.

## medaCy Model Management

One of medaCy's most powerful features is the ability to maintain, version and distribute medaCy compatible models with ease. The idea is simple - all the set-up code for a `Model` including a trained machine learning model is abstracted into an outside installable python package. This allows one to maintain the model with a version history just like any piece of software.

Once a model has been [packaged](packaging_a_medacy_model.md) and installed it can be used as follows:

```python
from medacy.model.model import Model

model = Model.load_external('medacy_model_clinical_notes')
annotations = model.predict("The patient took 5 mg of aspirin.")
```

See [Packaging a medaCy Model](packaging_a_medacy_model.md) for information on how to distribute your own trained models either internally or to the world.
