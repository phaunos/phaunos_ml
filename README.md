Tools for machine learning with audio data, including:

* audio activity detection (see [this notebook](https://bitbucket.org/securaxisteam/phaunos_ml/src/securaxis-tf2/notebooks/feature_extraction_and_activity_detection_example.ipynb?viewer=nbviewer))
* feature extraction: audio chunks or mel-spectrogram
* dataset management (train/test split, create subsets...)
* (de)serializing data

An example of a typical data pipeline, from the audio file to the batch of data to be fed to the model, is given [here](https://bitbucket.org/securaxisteam/phaunos_ml/src/securaxis-tf2/notebooks/data_pipeline_example.ipynb?viewer=nbviewer).

# Install

```
$ git clone git@bitbucket.org:securaxisteam/nsb_aad.git
$ git clone git@bitbucket.org:securaxisteam/phaunos_ml.git
$ cd /path/to/phaunos_ml
$ git checkout securaxis-tf2
$ export PYTHONPATH=$PYTHONPATH:/path/to/nsb_aad:/path/to/phaunos_ml
$ conda create -n phaunos_ml python=3.7
$ conda activate phaunos_ml
$ pip install -r requirements.txt
```
