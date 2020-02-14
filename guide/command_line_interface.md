# Using medaCy's Command Line Interface

MedaCy provides a command line interface (CLI) for using medaCy's core functionality. The core functionality includes 
cross validation, model training, and directory-level prediction, but other modules within medaCy also provide command 
line interfaces. While instructions can be displayed with the command `python -m medacy --help`, 
this guide provides a more thorough set of instructions.

A command to cross validate using the ClinicalPipeline might look like this:

```bash
(medacy_venv) $ python -m medacy -pl ClinicalPipeline -d ./your/dataset validate -k 7 -gt ./ground
```

Note that the pipeline and dataset are specified before the validate command, and that some arguments follow the 
name of the command. The first step to using the CLI is to identify which arguments must come before the command
and which must come after

## Global Arguments

Arguments are global when they could be applied to more than one command. 

### Dataset Arguments

* `-d` specifies the path to a dataset directory. If you are cross validating or fitting, this is the dataset of txt and ann files
 that will be used for that process. If you are predicting, this is the directory of txt files that will be predicted over.
* `-ent` specifies the path to a JSON with an "entities" key. This key's value should be a list of entities that are a 
subset of what appear in the selected dataset. If this argument is not used, medaCy will automatically use all the entities
that appear in the specified dataset.

### Pipeline Arguments
 
Two arguments relate specifically to the pipeline selection; only one should be used in a given command:
* `-pl` specifies which pre-configured pipeline to use. Options include `ClinicalPipeline`, `SystematicReviewPipeline`, 
`BertPipeline`, and `LstmSystematicReviewPipeline`.
* `-cpl` specifies the path to a JSON from which a custom pipeline can be constructed, see [here](creating_custom_pipeline_from_json.md) for a guide
on creating custom pipelines with JSON files

Note that `-ent` and `-cpl` both require JSON files. These can be the same JSON.

### Learner Arguments

While pipelines typically pre-configure the learners that they use, the BiLSTM and BERT learner allow for some arguments to be specified
from the command line.

These commands relate specifically to learning algorithms. They can be specified regardless of which pipeline or 
learner has been selected, but they will only be used if the pipeline uses a learner that supports that parameter, otherwise
they will simply be ignored.

* `-c` specifies which GPU device to use; if this is not specified, the command will run from the CPU, which may result
in slower performance for the BiLSTM and BERT learners. The authors of medaCy recommend the Python module [gpustat](https://pypi.org/project/gpustat/) for checking GPU availability.
* `-w` specifies a path to a word embedding binary file. This is required for the BiLSTM and is not used by any other learner.

The following arguments have default values but can be adjusted if you have knowledge of how changing them might affect
the learning process:
* `-b` specifies a batch size.
* `-e` specifies a number of epochs.
* `-pm` specifies a pretrained model for the BERT learner.
* `-crf` causes the BERT learner to use a CRF layer when this flag is present; it does not take an argument.
* `-lr` specifies a learning rate

### Logging and Testing Arguments

* `-t` runs the command in testing mode, which will only use one file in the specified dataset and use more verbose logging.
* `-lf` specifies a log file. The default log file is `medacy_n.log`, where `n` is the GPU being used or `cpu`. 
This argument allows you to specify a different log file.
* `-lc` causes the logging information to also be printed to the console.

## `validate`

If you don't have a pretrained medaCy model, the first step to using medaCy is creating or selecting a pipeline
configuration and validating it with your annotated dataset.

* `-k` specifies a number of folds. MedaCy uses sentence-level stratification. 10 is used by default, but a lower number will run more quickly.
* `-gt` specifies a directory to store the groundtruth version of the dataset. These are the ann files where sentence spans are aligned with which tokens are assigned a given
label within medaCy. This is not guaranteed to be the same as how they appear in the dataset itself because of how the selected pipeline tokenizes the data.
* `-pd` specifies a directory in which to store the predictions made throughout the cross validation process.

Note that the directories for `-gt` and `-pd` must already exist. If these arguments are not passed, the cross validation groundtruth and predictions will not be written anywhere.

The data outputted to the `-gt` and `-pd` directories is especially useful when combined with medaCy's inter-dataset 
agreement calculator. The command for this is `python -m medacy.tools.calculators.inter_dataset_agreement`.
Inter-dataset agreement can measure how similar the groudtruth is to the original dataset, or how similar the fold predictions
are to the groundtruth, etc.

A validate command might look like one of these:

```bash
(medacy_venv) $ python -m medacy -d ~/datasets/my_dataset -cpl ./my_pipeline.json validate -k 7 -gt ./ground -pd ./pred
(medacy_venv) $ python -m medacy -d ~/datasets/my_dataset -pl LstmSystematicReviewPipeline -c 0 -w ~/datasets/my_word_embeddings.bin validate
```

## `train`

There is little that needs to be specified here because the model, dataset, and pipeline arguments
are specified in the first section.

* `-f` specifies the location to write the model binary to.
* `-gt` specifies a directory to write the groundtruth to, the same as for `validate`.

```bash
(medacy_venv) $ python -m medacy -d ~/datasets/my_dataset -cpl ./my_pipeline.json train -gt ./ground -f ./my_crf_model.pkl
(medacy_venv) $ python -m medacy -d ~/datasets/my_dataset -pl LstmSystematicReviewPipeline -c 0 -w ~/datasets/my_word_embeddings.bin train -f ./my_bilstm_model.pkl
```

## `predict`

When predicting, one must specify in the first section which pipeline they want to use and which dataset of txt files to predict over.

* `-m` specifies the path to the model binary file to use. Remember to use the same pipeline that was used to create the model binary selected here.
* `-pd` specifies the directory to write the predictions to. 

```bash
(medacy_venv) $ python -m medacy -d ~/datasets/my_txt_files -cpl ./my_pipeline.json predict -m ./my_crf_model.pkl -pd ./crf_predictions
(medacy_venv) $ python -m medacy -d ~/datasets/my_txt_files -pl LstmSystematicReviewPipeline -w ~/datasets/my_word_embeddings.bin predict -m ./my_bilstm_model.pkl -pd ./bilstm_predictions
```
