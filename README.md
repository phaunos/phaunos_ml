Tools for machine learning with audio data, including:

* audio activity detection (see [this notebook](https://bitbucket.org/securaxisteam/phaunos_ml/src/securaxis/notebooks/feature_extraction_and_segmentation.ipynb?viewer=nbviewer))
* feature extraction: audio chunks or mel-spectrogram
* dataset management (train/test split, create subsets...)
* saving features as TFRecord files

Some examples of the whole data pipeline for the BirdCLEF bird recognition (single-label) and the Freesound audio tagging (multi-label) 2019 challenges are given [here](https://bitbucket.org/securaxisteam/phaunos_ml/src/securaxis/notebooks).

# Install

```
$ git clone git@bitbucket.org:securaxisteam/nsb_aad.git
$ git clone git@bitbucket.org:securaxisteam/phaunos_ml.git
$ export PYTHONPATH=$PYTHONPATH:/path/to/nsb_aad:/path/tp/phaunos_ml
$ conda create -n phaunos_ml python=3.6
$ conda activate phaunos_ml
$ pip install -r requirements_{cpu,gpu}.txt
```
