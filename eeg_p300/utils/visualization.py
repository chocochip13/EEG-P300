"""
Utilities for data visualization and plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

def plot_within_aucs(aucpath, filepath, title):
    """
    Plot AUCs for within-subject evaluation.
    
    Parameters
    ----------
    aucpath : str
        Path to directory containing AUC files
    filepath : str
        Path to save the plot
    title : str
        Title for the plot
        
    Returns
    -------
    tuple
        Tuple containing (mean_auc, std_auc)
    """
    logger.info(f"Plotting within-subject AUCs from {aucpath}")
    data_to_plot = []
    aucs = np.zeros((8, 50))
    
    for i in range(8):
        aucs[i, :] = np.loadtxt(f"{aucpath}/s{i}_auc.npy")
        data_to_plot.append(np.loadtxt(f"{aucpath}/s{i}_auc.npy"))

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot, showmeans=True, meanline=True)
    plt.title(f"{title}\n Mean AUC: {np.mean(aucs):.2f}±{np.std(aucs):.2f}")
    plt.ylabel("AUC")
    plt.xlabel('Subjects')
    plt.grid(False)
    plt.savefig(filepath)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    logger.info(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return mean_auc, std_auc


def plot_cross_aucs(aucpath, filepath, title):
    """
    Plot AUCs for cross-subject evaluation.
    
    Parameters
    ----------
    aucpath : str
        Path to AUC file
    filepath : str
        Path to save the plot
    title : str
        Title for the plot
        
    Returns
    -------
    tuple
        Tuple containing (mean_auc, std_auc)
    """
    logger.info(f"Plotting cross-subject AUCs from {aucpath}")
    aucs = np.loadtxt(aucpath)

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8], aucs)
    plt.title(f"{title}\n Mean AUC: {np.mean(aucs):.2f}±{np.std(aucs):.2f}")
    plt.ylabel("AUC")
    plt.xlabel('Subjects')
    plt.grid(False)
    plt.savefig(filepath)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    logger.info(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return mean_auc, std_auc


def plot_training_history(history, filepath):
    """
    Plot training history.
    
    Parameters
    ----------
    history : keras.callbacks.History
        Training history
    filepath : str
        Path to save the plot
    """
    logger.info("Plotting training history")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot AUC if available
    if 'roc_auc' in history.history:
        ax2.plot(history.history['roc_auc'], label='Train')
        ax2.plot(history.history['val_roc_auc'], label='Validation')
        ax2.set_title('Model AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_eeg_signals(X, y, channels, n_samples=3, filepath=None):
    """
    Plot EEG signals for visualization.
    
    Parameters
    ----------
    X : ndarray
        EEG data with shape (n_samples, n_timepoints, n_channels)
    y : ndarray
        Labels with shape (n_samples,)
    channels : list
        List of channel names
    n_samples : int
        Number of samples to plot
    filepath : str, optional
        Path to save the plot
    """
    logger.info(f"Plotting {n_samples} EEG signal samples")
    n_channels = len(channels)
    
    # Get samples of target and non-target
    target_idx = np.where(y == 1)[0][:n_samples]
    non_target_idx = np.where(y == 0)[0][:n_samples]
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
    
    # Plot target signals
    for i in range(n_samples):
        for ch in range(n_channels):
            axes[i, 0].plot(X[target_idx[i], :, ch], label=channels[ch])
        axes[i, 0].set_title(f"Target Signal (Sample {i+1})")
        if i == 0:
            axes[i, 0].legend()
    
    # Plot non-target signals
    for i in range(n_samples):
        for ch in range(n_channels):
            axes[i, 1].plot(X[non_target_idx[i], :, ch], label=channels[ch])
        axes[i, 1].set_title(f"Non-Target Signal (Sample {i+1})")
        if i == 0:
            axes[i, 1].legend()
    
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
