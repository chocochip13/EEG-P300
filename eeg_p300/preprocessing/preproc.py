"""
Functions for preprocessing EEG signals data.
"""
import numpy as np
import mne
from mne import Epochs, find_events
import logging

logger = logging.getLogger(__name__)

def data_extract(dataset):
    """
    Extract data components from the raw dataset.
    
    Parameters
    ----------
    dataset : dict
        Raw dataset loaded from .mat file
        
    Returns
    -------
    Tuple
        Tuple containing (channels, X, y, y_stim, trial, classes, classes_stim, 
                         gender, age, ALSfrs, onsetALS)
    """
    logger.info("Extracting data components from raw dataset")
    data = dataset['data']
    channels_temp = (data[0][0][0]).flatten()
    channels = []
    for i in range(8):
        channels.append(channels_temp[i][0])
    X = data[0][0][1]
    y = data[0][0][2].flatten()
    y_stim = data[0][0][3]
    trial = data[0][0][4].flatten()
    classes = data[0][0][5]
    classes_stim = data[0][0][6]
    gender = data[0][0][7]
    age = data[0][0][8]
    ALSfrs = data[0][0][9]
    onsetALS = data[0][0][10]
    
    return channels, X, y, y_stim, trial, classes, classes_stim, gender, age, ALSfrs, onsetALS


def set_info(X, channels, sfreq=256):
    """
    Create MNE info object and raw array from EEG data.
    
    Parameters
    ----------
    X : ndarray
        EEG signal data with shape (n_samples, n_channels)
    channels : list
        List of channel names
    sfreq : int
        Sampling frequency in Hz
        
    Returns
    -------
    mne.io.RawArray
        Raw MNE object with EEG data
    """
    logger.info(f"Creating MNE info with {len(channels)} channels at {sfreq} Hz")
    info = mne.create_info(channels, sfreq, ch_types=['eeg']*len(channels))
    raw = mne.io.RawArray(X.T, info)
    
    raw.set_montage("standard_1020")
    return raw


def add_stim(raw, y):
    """
    Add stimulus channel to raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.RawArray
        Raw MNE object with EEG data
    y : ndarray
        Stimulus data to add as a channel
        
    Returns
    -------
    mne.io.RawArray
        Raw MNE object with stimulus channel added
    """
    logger.info("Adding stimulus channel to raw data")
    info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(np.reshape(y, (-1, 1)).T, info)
    raw.add_channels([stim_raw], force_update_info=True)
    return raw


def bp_filter(dataset, f_low=0.1, f_high=30, picks='eeg', iir_params=dict(order=8, ftype='butter')):
    """
    Apply bandpass filter to EEG data.
    
    Parameters
    ----------
    dataset : dict
        Raw dataset loaded from .mat file
    f_low : float
        Low cutoff frequency in Hz
    f_high : float
        High cutoff frequency in Hz
    picks : str or list
        Channels to filter
    iir_params : dict
        Parameters for IIR filter
        
    Returns
    -------
    mne.io.RawArray
        Filtered raw MNE object
    """
    logger.info(f"Applying bandpass filter: {f_low}-{f_high} Hz")
    channels, X, y, y_stim, trial, classes, classes_stim, gender, age, ALSfrs, onsetALS = data_extract(dataset)
    raw = set_info(X, channels, sfreq=256)
    raw = add_stim(raw, y)
    filt_raw = raw.copy().filter(f_low, f_high, picks=picks, method='iir', iir_params=iir_params, verbose=True)
    return filt_raw


def epochs_gen(filt_raw, event_id=dict(NT=1, T=2), tmin=-0.2, tmax=0.8, picks='eeg'):
    """
    Generate epochs from filtered raw data.
    
    Parameters
    ----------
    filt_raw : mne.io.RawArray
        Filtered raw MNE object
    event_id : dict
        Dictionary mapping event names to event ids
    tmin : float
        Start time of the epochs in seconds
    tmax : float
        End time of the epochs in seconds
    picks : str or list
        Channels to include in epochs
        
    Returns
    -------
    mne.Epochs
        Epochs object
    """
    logger.info(f"Generating epochs from {tmin} to {tmax} seconds around events")
    events = find_events(filt_raw)
    epochs = Epochs(filt_raw, events, event_id, tmin, tmax, picks=picks, baseline=(-0.2, 0), 
                    reject=None, preload=True)
    return epochs
