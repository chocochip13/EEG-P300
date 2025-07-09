#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train and evaluate EEG P300 models.
"""
import os
import numpy as np
import tensorflow as tf
import logging
import sys
import argparse
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eeg_p300.config import Config
from eeg_p300.utils.data_utils import load_db
from eeg_p300.evaluation.cross_subject import evaluate_cross_subject_model
from eeg_p300.evaluation.within_subject import evaluate_within_subject_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_environment(config):
    """
    Set up the environment for training.
    
    Parameters
    ----------
    config : Config
        Configuration object
    """
    # Set random seeds for reproducibility
    np.random.seed(config.model.random_seed)
    tf.random.set_seed(config.model.random_seed)
    
    # Set up TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=debug, 1=info, 2=warning, 3=error


def train_and_evaluate(config, evaluation_type="within", model_type="sepconv1d"):
    """
    Train and evaluate P300 models.
    
    Parameters
    ----------
    config : Config
        Configuration object
    evaluation_type : str
        Type of evaluation ('within' or 'cross')
    model_type : str
        Type of model ('sepconv1d' or 'cnn1')
        
    Returns
    -------
    dict
        Dictionary containing evaluation results
    """
    # Load data
    data_path = Path(config.data.processed_data_path)
    data_file = data_path / "A-epo.npy"
    labels_file = data_path / "A-labels.npy"
    
    if not data_file.exists() or not labels_file.exists():
        logger.error(f"Data files not found. Run prepare_data.py first.")
        return None
        
    logger.info(f"Loading data from {data_file} and {labels_file}")
    data, labels = load_db(str(data_file), str(labels_file))
    logger.info(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Reshape data if needed for cross-subject evaluation
    if data.ndim == 3:  # If data is (n_samples, n_timepoints, n_channels)
        logger.warning("Reshaping data for subject-wise evaluation")
        n_subjects = 8  # Number of subjects in the P300 Speller database
        n_samples_per_subject = data.shape[0] // n_subjects
        data = data.reshape(n_subjects, n_samples_per_subject, data.shape[1], data.shape[2])
        labels = labels.reshape(n_subjects, n_samples_per_subject)
    
    # Choose evaluation strategy
    if evaluation_type.lower() == "cross":
        logger.info("Using cross-subject evaluation")
        results = evaluate_cross_subject_model(data, labels, config, model_type=model_type)
    else:
        logger.info("Using within-subject evaluation")
        results = evaluate_within_subject_models(data, labels, config, model_type=model_type)
    
    return results


def main():
    """Main function for model training and evaluation."""
    parser = argparse.ArgumentParser(description='Train and evaluate EEG P300 models')
    parser.add_argument('--model', type=str, choices=['sepconv1d', 'cnn1'], default='sepconv1d',
                        help='Type of model to train (default: sepconv1d)')
    parser.add_argument('--eval-type', type=str, choices=['within', 'cross'], default='within',
                        help='Type of evaluation to perform (default: within)')
    parser.add_argument('--data-path', type=str, help='Path to processed data directory')
    parser.add_argument('--results-path', type=str, help='Path to results directory')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--filters', type=int, help='Number of filters (for sepconv1d)')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override configuration with command-line arguments if provided
    if args.data_path:
        config.data.processed_data_path = args.data_path
    if args.results_path:
        config.data.results_path = args.results_path
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.epochs:
        config.model.epochs = args.epochs
    if args.filters:
        config.model.filters = args.filters
    if args.patience:
        config.model.patience = args.patience
    
    # Set up environment
    setup_environment(config)
    
    # Train and evaluate
    model_type = args.model
    evaluation_type = args.eval_type
    
    logger.info(f"Starting training and evaluation with {model_type} model using {evaluation_type}-subject evaluation")
    results = train_and_evaluate(config, evaluation_type=evaluation_type, model_type=model_type)
    
    if results:
        logger.info(f"Results: Mean AUC = {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        logger.info("Training and evaluation completed successfully")
    else:
        logger.error("Training and evaluation failed")


if __name__ == "__main__":
    main()
