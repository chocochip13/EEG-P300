"""
Cross-subject evaluation for EEG P300 models.
"""
import numpy as np
import time
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import logging
from ..utils.data_utils import EEGChannelScaler
from ..utils.visualization import plot_cross_aucs
from ..models.sepconv1d import SepConv1D
from ..models.cnn1 import CNN1
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

def evaluate_cross_subject_model(data, labels, config, model_type="sepconv1d"):
    """
    Train and evaluate model for each subject in the P300 Speller database
    using leave-one-group-out cross validation.
    
    Parameters
    ----------
    data : ndarray
        EEG data with shape (n_subjects, n_trials, n_samples, n_channels)
    labels : ndarray
        Labels with shape (n_subjects, n_trials)
    config : Config
        Configuration object
    model_type : str
        Model type to use ('sepconv1d' or 'cnn1')
        
    Returns
    -------
    dict
        Dictionary containing evaluation results
    """
    logger.info("Starting cross-subject evaluation")
    logger.info(f"Using model: {model_type}")
    
    n_sub = data.shape[0]
    n_ex_sub = data.shape[1]
    n_samples = data.shape[2]
    n_channels = data.shape[3]

    aucs = np.zeros(n_sub)
    inf_time = np.zeros(n_sub)

    # Reshape data for leave-one-subject-out cross-validation
    data_flat = data.reshape((n_sub * n_ex_sub, n_samples, n_channels))
    labels_flat = labels.reshape((n_sub * n_ex_sub))
    groups = np.array([i for i in range(n_sub) for j in range(n_ex_sub)])

    # Initialize cross-validation
    cv = LeaveOneGroupOut()
    
    # Loop through each subject (left out as test set)
    for k, (t, v) in enumerate(cv.split(data_flat, labels_flat, groups)):
        X_train, y_train = data_flat[t], labels_flat[t]
        X_test, y_test = data_flat[v], labels_flat[v]
        
        # Select validation set from training subjects
        rg = np.random.choice(t, 1)
        sv = groups[t] == groups[rg]
        st = np.logical_not(sv)
        X_train, y_train = data_flat[t][st], labels_flat[t][st]
        X_valid, y_valid = data_flat[t][sv], labels_flat[t][sv]
        
        logger.info(f"Partition {k}: train = {X_train.shape}, valid = {X_valid.shape}, test = {X_test.shape}")
        logger.info(f"Groups train = {np.unique(groups[t][st])}, valid = {np.unique(groups[t][sv])}, test = {np.unique(groups[v])}")

        # Channel-wise feature standardization
        sc = EEGChannelScaler(n_channels=n_channels)
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

        # Create model
        if model_type.lower() == "sepconv1d":
            model = SepConv1D(Chans=n_channels, Samples=n_samples, Filters=config.model.filters)
        elif model_type.lower() == "cnn1":
            model = CNN1(Chans=n_channels, Samples=n_samples)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        logger.info(model.summary())
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Early stopping
        es = EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=config.model.patience, 
            restore_best_weights=True
        )

        # Train model
        start_train = time.time()
        history = model.fit(
            X_train,
            y_train,
            batch_size=config.model.batch_size,
            epochs=config.model.epochs,
            validation_data=(X_valid, y_valid),
            callbacks=[es]
        )
        train_time = time.time() - start_train
        logger.info(f"Training time: {train_time:.2f} seconds")

        # Evaluate model
        start_test = time.time()
        proba_test = model.predict(X_test)
        test_time = time.time() - start_test

        test_size = X_test.shape[0]
        inf_time[k] = test_time / test_size

        aucs[k] = roc_auc_score(y_test, proba_test)
        logger.info(f'Subject {k} -- AUC: {aucs[k]:.4f}')
        
        # Save model and data if needed
        model_path = f"{config.data.results_path}/cross_subject/s{k}_model.h5"
        model.save_weights(model_path)
        logger.info(f"Saved model weights to {model_path}")
        
        # Clear Keras session
        K.clear_session()

    # Save results
    results_path = f"{config.data.results_path}/cross_subject"
    np.savetxt(f"{results_path}/aucs.npy", aucs)
    np.savetxt(f"{results_path}/inf_time.npy", inf_time)
    
    # Plot results
    plot_path = f"{results_path}/{model_type}_cross_auc.png"
    mean_auc, std_auc = plot_cross_aucs(
        f"{results_path}/aucs.npy", 
        plot_path, 
        f"{model_type.upper()} - Cross"
    )
    
    # Return results
    results = {
        "aucs": aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "inference_times": inf_time,
        "mean_inference_time": np.mean(inf_time)
    }
    
    return results
