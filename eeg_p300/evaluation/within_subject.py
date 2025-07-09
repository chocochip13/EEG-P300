"""
Within-subject evaluation for EEG P300 models.
"""
import numpy as np
import time
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import logging
from ..utils.data_utils import EEGChannelScaler
from ..utils.visualization import plot_within_aucs
from ..models.sepconv1d import SepConv1D
from ..models.cnn1 import CNN1
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

def evaluate_subject_model(data, labels, config, subject, model_type="sepconv1d"):
    """
    Train and evaluate model for a specific subject in the P300 Speller database
    using repeated stratified K-fold cross validation.
    
    Parameters
    ----------
    data : ndarray
        EEG data with shape (n_subjects, n_trials, n_samples, n_channels)
    labels : ndarray
        Labels with shape (n_subjects, n_trials)
    config : Config
        Configuration object
    subject : int
        Subject index to evaluate
    model_type : str
        Model type to use ('sepconv1d' or 'cnn1')
        
    Returns
    -------
    dict
        Dictionary containing evaluation results
    """
    logger.info(f"Evaluating subject {subject} using {model_type} model")
    
    n_trials = data.shape[1]
    n_samples = data.shape[2]
    n_channels = data.shape[3]
    
    n_folds = config.evaluation.cross_validation_folds
    n_repeats = config.evaluation.cross_validation_repeats
    
    inf_time = np.zeros(n_folds * n_repeats)
    aucs = np.zeros(n_folds * n_repeats)

    # Initialize cross-validation
    cv = RepeatedStratifiedKFold(
        n_splits=n_folds, 
        n_repeats=n_repeats, 
        random_state=config.evaluation.random_seed
    )
    
    # Loop through each fold
    for k, (t, v) in enumerate(cv.split(data[subject], labels[subject])):
        X_train, y_train = data[subject, t, :, :], labels[subject, t]
        X_test, y_test = data[subject, v, :, :], labels[subject, v]
        
        # Split training data into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train, 
            test_size=config.evaluation.test_size, 
            shuffle=True, 
            random_state=config.evaluation.random_seed
        )
        
        logger.info(f'Partition {k}: X_train = {X_train.shape}, X_valid = {X_valid.shape}, X_test = {X_test.shape}')

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
        
        if k == 0:  # Only log the model summary once
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
        logger.info(f'Subject {subject}, Partition {k} -- AUC: {aucs[k]:.4f}')
        
        # Clear Keras session to free memory
        K.clear_session()

    # Save results
    results_path = f"{config.data.results_path}/within_subject"
    np.savetxt(f"{results_path}/s{subject}_auc.npy", aucs)
    
    # Return fold results
    return {
        "subject": subject,
        "aucs": aucs,
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "inference_times": inf_time,
        "mean_inference_time": np.mean(inf_time)
    }


def evaluate_within_subject_models(data, labels, config, model_type="sepconv1d"):
    """
    Train and evaluate models for all subjects using within-subject evaluation.
    
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
    logger.info(f"Starting within-subject evaluation using {model_type} model")
    
    n_subjects = data.shape[0]
    results = []
    
    # Evaluate each subject
    for subject in range(n_subjects):
        subject_results = evaluate_subject_model(
            data, 
            labels, 
            config, 
            subject, 
            model_type=model_type
        )
        results.append(subject_results)
    
    # Compile overall results
    all_aucs = np.array([r["mean_auc"] for r in results])
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    
    # Save and plot overall results
    results_path = f"{config.data.results_path}/within_subject"
    plot_path = f"{results_path}/{model_type}_within_auc.png"
    plot_within_aucs(
        results_path, 
        plot_path, 
        f"{model_type.upper()} - Within"
    )
    
    # Return overall results
    overall_results = {
        "subject_results": results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "all_subjects_mean_aucs": all_aucs
    }
    
    logger.info(f"Overall mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return overall_results
