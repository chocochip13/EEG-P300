## Introduction
This project contains works on processing raw EEG signals data and prediction of P300 signals. The aim of this project is to:

* [Preprocessing](https://github.com/chocochip13/EEG-P300/blob/master/notebooks/preproc.py)
* [Exploratory Data Analysis](https://github.com/chocochip13/EEG-P300/blob/master/notebooks/EDA.ipynb)
* Build Prediction model for P300

## Dataset
The EEG signals dataset is the P300 speller with ALS patients from [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets)
datasets. 
This dataset contains 8 subjects diagnosed with ALS, the dataset has 8 channels with sampling frequency 256Hz.
More details about the dataset can be found [here](https://lampx.tugraz.at/~bci/database/008-2014/description.pdf).

## Preprocessing
[preproc.py](https://github.com/chocochip13/EEG-P300/blob/master/notebooks/preproc.py) contains the functions for preprocessing.
The steps for working and preprocessing the EEG signals data are:
* Extract the dataset
* Set info
* Find/ Add stimulus channel
* Filter dataset
