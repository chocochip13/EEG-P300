"""
Utility functions for data handling and processing.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def load_db(datafile, labelsfile):
    """
    Load P300 Speller database.
    
    Parameters
    ----------
    datafile : str
        Path to data file (.npy)
    labelsfile : str
        Path to labels file (.npy)
        
    Returns
    -------
    tuple
        Tuple containing (data, labels)
    """
    logger.info(f"Loading data from {datafile} and {labelsfile}")
    data = np.load(datafile)
    labels = np.load(labelsfile)
    logger.info(f"Data shape = {data.shape}, Labels shape = {labels.shape}")
    
    return data, labels


class EEGChannelScaler:
    """
    Class to scale each channel of an EEG signal separately.
    """
    def __init__(self, n_channels=6):
        """
        Initialize a standard scaler for each channel.
        
        Parameters
        ----------
        n_channels : int
            Number of EEG channels
        """
        self.n_channels_ = n_channels
        self.sc_ = []
        for c in range(self.n_channels_):
            self.sc_.append(StandardScaler())
        self.fitted_ = False
            
    def fit_transform(self, X):
        """
        Fit the standard scaler for each channel using the training data.
        
        Parameters
        ----------
        X : ndarray
            EEG data with shape (n_samples, n_timepoints, n_channels)
            
        Returns
        -------
        ndarray
            Scaled EEG data
        """
        if X.shape[2] != self.n_channels_:
            error_msg = f'Error: Expected {self.n_channels_} channels, got {X.shape[2]} instead.'
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Fitting and transforming data with shape {X.shape}")
        for c in range(self.n_channels_):
            X[:, :, c] = self.sc_[c].fit_transform(X[:, :, c])

        self.fitted_ = True
        
        return X
    
    def transform(self, X):
        """
        Scale an array for each channel separately.
        
        Parameters
        ----------
        X : ndarray
            EEG data with shape (n_samples, n_timepoints, n_channels)
            
        Returns
        -------
        ndarray
            Scaled EEG data
        """
        logger.info(f"Transforming data with shape {X.shape}")
        for c in range(self.n_channels_):
            X[:, :, c] = self.sc_[c].transform(X[:, :, c])

        return X


def make_trial_average(X, y, n_trials=2, pos_samples=10000, neg_samples=10000):
    """
    Generate trial averages for data augmentation.
    
    Parameters
    ----------
    X : ndarray
        EEG data with shape (n_samples, n_timepoints, n_channels)
    y : ndarray
        Labels with shape (n_samples,)
    n_trials : int
        Number of trials to average
    pos_samples : int
        Number of positive samples to generate
    neg_samples : int
        Number of negative samples to generate
        
    Returns
    -------
    tuple
        Tuple containing (averaged_data, averaged_labels)
    """
    logger.info(f'Generating {pos_samples} positive and {neg_samples} negative samples of {n_trials}-trial averages')

    X_pos = X[y == 1, :, :]
    X_neg = X[y == 0, :, :]
    
    X_avg = np.zeros((pos_samples + neg_samples, X.shape[1], X.shape[2]))
    y_avg = np.zeros(pos_samples + neg_samples)
    y_avg[:pos_samples] = 1
    
    for i in range(pos_samples):
        pos_trials = np.random.choice(X_pos.shape[0], n_trials)
        X_avg[i, :, :] = np.mean(X_pos[pos_trials, :, :], axis=0)

    for i in range(neg_samples):
        neg_trials = np.random.choice(X_neg.shape[0], n_trials)
        X_avg[pos_samples + i, :, :] = np.mean(X_neg[neg_trials, :, :], axis=0)
            
    perm = np.random.permutation(pos_samples + neg_samples)
    X_avg = X_avg[perm, :, :]
    y_avg = y_avg[perm]
        
    return X_avg, y_avg


def stack_trials(X, y, n_trials=2, pos_samples=1000, neg_samples=1000):
    """
    Stack multiple trials for ensemble learning.
    
    Parameters
    ----------
    X : ndarray
        EEG data with shape (n_samples, n_timepoints, n_channels)
    y : ndarray
        Labels with shape (n_samples,)
    n_trials : int
        Number of trials to stack
    pos_samples : int
        Number of positive samples to generate
    neg_samples : int
        Number of negative samples to generate
        
    Returns
    -------
    tuple
        Tuple containing (stacked_data, stacked_labels)
    """
    logger.info(f'X shape = {X.shape}, y shape = {y.shape}')
    logger.info(f'Generating {pos_samples} positive and {neg_samples} negative samples of {n_trials}-trial stacks')
    
    X_pos = X[y == 1, :, :]
    X_neg = X[y == 0, :, :]
    
    X_stack = np.zeros((pos_samples + neg_samples, X.shape[1], X.shape[2], n_trials))
    y_stack = np.zeros(pos_samples + neg_samples)
    
    for i in range(pos_samples):
        pos_trials = np.random.choice(X_pos.shape[0], n_trials)
        X_stack[i, :, :, :] = X_pos[pos_trials, :, :].transpose(1, 2, 0)
        y_stack[i] = 1
        
    for i in range(neg_samples):
        neg_trials = np.random.choice(X_neg.shape[0], n_trials)
        X_stack[pos_samples + i, :, :, :] = X_neg[neg_trials, :, :].transpose(1, 2, 0)
        y_stack[pos_samples + i] = 0
            
    perm = np.random.permutation(pos_samples + neg_samples)
    X_stack = X_stack[perm, :, :, :]
    y_stack = y_stack[perm]
            
    return X_stack, y_stack


def balance_data(X, y, n_samples=1000, btype='downsample'):
    """
    Balance dataset by downsampling or upsampling.
    
    Parameters
    ----------
    X : ndarray
        Data array
    y : ndarray
        Labels array
    n_samples : int
        Number of samples for balancing
    btype : str
        Balancing type ('downsample' or 'upsample')
        
    Returns
    -------
    tuple
        Tuple containing (balanced_data, balanced_labels)
    """
    from sklearn.utils import resample
    
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    if btype == 'downsample':
        logger.info(f"Downsampling negative class to {n_samples} samples")
        X_neg = resample(X_neg, replace=False, n_samples=n_samples)
    else:
        logger.info(f"Upsampling positive class to {n_samples} samples")
        X_pos = resample(X_pos, replace=True, n_samples=n_samples)

    X_balanced = np.concatenate((X_neg, X_pos))
    y_balanced = np.zeros(X_neg.shape[0] + X_pos.shape[0])
    y_balanced[X_neg.shape[0]:] = 1
    perm = np.random.permutation(X_balanced.shape[0])
    
    return X_balanced[perm], y_balanced[perm]
