Tools for machine learning with audio data, including:
* audio activity detection (see [this notebook](/src/securaxis/notebooks/feature_extraction_and_segmentation.ipynb?viewer=nbviewer))
* feature extraction: audio chunks or mel-spectrogram
* dataset management (train/test split, create subsets...)
* saving features as TFRecord files

Some examples of the whole data pipeline for the BirdCLEF bird recognition (single-label) and the Freesound audio tagging (multi-label) 2019 challenges are given [here](/src/securaxis/notebooks).

# Install

```
$ conda create -n phaunos_ml python=3.6
$ pip install git+https://github.com/phaunos/phaunos_ml
```
