Tools for machine learning with audio data, including:
- feature extraction (only mel spectrogram for now)
- dataset management (train/test split, create subsets...)
- saving features as TFRecord files

Some examples of the whole data pipeline for the BirdCLEF bird recognition (single-label) and the Freesound audio tagging (multi-label) 2019 challenges are given [here](https://github.com/phaunos/phaunos_ml/tree/master/notebooks).

#Install

```
$ conda create -n phaunos_ml python=3.6
$ pip install git+https://github.com/phaunos/phaunos_ml
```
