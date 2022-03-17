import numpy as np
import mne
from mne import Epochs, find_events


def data_extract(dataset):
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
    info = mne.create_info(channels, sfreq, ch_types=['eeg']*8)
    raw = mne.io.RawArray(X.T, info)
    
    raw.set_montage("standard_1020")
    return raw


def add_stim(raw, y):
    info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(np.reshape(y, (-1, 1)).T, info)
    raw.add_channels([stim_raw], force_update_info=True)
    return raw

#TODO: Remove other information
def bp_filter(dataset, f_low=0.1, f_high=30, picks='eeg', iir_params= dict(order=8, ftype='butter')):
    channels, X, y, y_stim, trial, classes, classes_stim, gender, age, ALSfrs, onsetALS = data_extract(dataset)
    raw = set_info(X, channels, sfreq=256)
    raw = add_stim(raw, y)
    filt_raw = raw.copy().filter(f_low, f_high, picks=picks, method='iir', iir_params=iir_params, verbose=True)
    # plot_raw = raw.plot(color = 'blue', scalings= 'auto',duration =10 ,start = 8,
    #                     title = "Raw EEG signals", show_scalebars=True, show_scrollbars=True)
    # plot_filt = filt_raw.plot(color = 'blue', scalings = 'auto', duration =10 ,start = 8)
    return filt_raw

#TODO: Remove events
def epochs_gen(filt_raw, event_id=dict(NT=1, T=2), tmin=-0.2, tmax=0.8, picks='eeg'):
    # filt_raw = bp_filter(dataset, f_low = 0.1, f_high = 30, picks = 'eeg', iir_params = dict(order=8, ftype='butter')
    events = find_events(filt_raw)
    epochs = Epochs(filt_raw, events, event_id, tmin, tmax, picks=picks, baseline=(-0.2, 0), reject=None, preload=True)
    return epochs
