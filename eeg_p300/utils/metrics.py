"""
Utility functions for model evaluation and metrics.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class ROCCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to calculate ROC curve and AUC during training.
    """
    
    def __init__(self, training_data, validation_data):
        """
        Initialize the callback with training and validation data.
        
        Parameters
        ----------
        training_data : tuple
            Tuple containing (X_train, y_train)
        validation_data : tuple
            Tuple containing (X_val, y_val)
        """
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate and log AUC at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : dict
            Dictionary of logs from the model
        """
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        
        logs = logs or {}
        logs['roc_auc'] = roc
        logs['val_roc_auc'] = roc_val
        
        logger.info(f'Epoch {epoch+1}: AUC Train: {roc:.4f} AUC Valid: {roc_val:.4f}')
        
        return


def calculate_aucs(model, data_list, labels_list):
    """
    Calculate AUC for multiple datasets.
    
    Parameters
    ----------
    model : keras.Model
        Trained model
    data_list : list
        List of data arrays
    labels_list : list
        List of label arrays
        
    Returns
    -------
    list
        List of AUC values for each dataset
    """
    n_datasets = len(data_list)
    aucs = np.zeros(n_datasets)
    
    for i in range(n_datasets):
        proba = model.predict(data_list[i])
        aucs[i] = roc_auc_score(labels_list[i], proba)
        logger.info(f'Dataset {i+1}: AUC = {aucs[i]:.4f}')
        
    return aucs
