#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to visualize EEG data and model results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import argparse
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eeg_p300.config import Config
from eeg_p300.utils.data_utils import load_db
from eeg_p300.utils.visualization import (
    plot_within_aucs, 
    plot_cross_aucs, 
    plot_eeg_signals
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def visualize_eeg_data(config):
    """
    Visualize EEG signals from the dataset.
    
    Parameters
    ----------
    config : Config
        Configuration object
    """
    # Load data
    data_path = Path(config.data.processed_data_path)
    data_file = data_path / "A-epo.npy"
    labels_file = data_path / "A-labels.npy"
    
    if not data_file.exists() or not labels_file.exists():
        logger.error(f"Data files not found. Run prepare_data.py first.")
        return False
        
    logger.info(f"Loading data from {data_file} and {labels_file}")
    data, labels = load_db(str(data_file), str(labels_file))
    logger.info(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # If data is in shape (n_subjects, n_trials, n_samples, n_channels)
    # Reshape to (n_samples, n_timepoints, n_channels) for visualization
    if data.ndim == 4:
        n_subjects = data.shape[0]
        n_trials_per_subject = data.shape[1]
        data = data.reshape(-1, data.shape[2], data.shape[3])
        labels = labels.reshape(-1)
    
    # Define channel names (these may need to be adjusted based on your dataset)
    channel_names = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']
    
    # Create output directory for plots
    plots_dir = Path(config.data.results_path) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot example EEG signals
    logger.info("Plotting example EEG signals")
    plot_eeg_signals(
        data, 
        labels, 
        channel_names, 
        n_samples=3, 
        filepath=str(plots_dir / "eeg_signals.png")
    )
    
    # Plot signals for each subject if data is available in subject format
    if data.ndim == 4:
        for i in range(n_subjects):
            logger.info(f"Plotting EEG signals for subject {i+1}")
            plot_eeg_signals(
                data[i], 
                labels[i], 
                channel_names, 
                n_samples=2, 
                filepath=str(plots_dir / f"subject_{i+1}_signals.png")
            )
    
    return True


def visualize_model_results(config, model_type="sepconv1d"):
    """
    Visualize model evaluation results.
    
    Parameters
    ----------
    config : Config
        Configuration object
    model_type : str
        Type of model ('sepconv1d' or 'cnn1')
    """
    # Create output directory for plots
    plots_dir = Path(config.data.results_path) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize within-subject results if available
    within_results_path = Path(config.data.results_path) / "within_subject"
    if within_results_path.exists():
        logger.info("Visualizing within-subject evaluation results")
        try:
            plot_within_aucs(
                str(within_results_path),
                str(plots_dir / f"{model_type}_within_auc.png"),
                f"{model_type.upper()} - Within"
            )
            logger.info(f"Saved within-subject plot to {plots_dir / f'{model_type}_within_auc.png'}")
        except Exception as e:
            logger.error(f"Error plotting within-subject results: {e}")
    
    # Visualize cross-subject results if available
    cross_results_path = Path(config.data.results_path) / "cross_subject"
    if cross_results_path.exists() and (cross_results_path / "aucs.npy").exists():
        logger.info("Visualizing cross-subject evaluation results")
        try:
            plot_cross_aucs(
                str(cross_results_path / "aucs.npy"),
                str(plots_dir / f"{model_type}_cross_auc.png"),
                f"{model_type.upper()} - Cross"
            )
            logger.info(f"Saved cross-subject plot to {plots_dir / f'{model_type}_cross_auc.png'}")
        except Exception as e:
            logger.error(f"Error plotting cross-subject results: {e}")
    
    # Compare models if both CNN1 and SepConv1D results are available
    if model_type == "all":
        try:
            # Compare within-subject results
            if (within_results_path / "s0_auc.npy").exists():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for model in ["cnn1", "sepconv1d"]:
                    aucs = []
                    for i in range(8):  # 8 subjects
                        subj_file = within_results_path / f"s{i}_auc.npy"
                        if subj_file.exists():
                            subj_aucs = np.loadtxt(str(subj_file))
                            aucs.append(np.mean(subj_aucs))
                    
                    if aucs:
                        ax.bar(
                            np.arange(len(aucs)) + (0.2 if model == "cnn1" else -0.2), 
                            aucs, 
                            width=0.4, 
                            label=model.upper()
                        )
                
                ax.set_xlabel("Subject")
                ax.set_ylabel("Mean AUC")
                ax.set_title("Model Comparison - Within Subject")
                ax.set_xticks(np.arange(8))
                ax.set_xticklabels([f"S{i+1}" for i in range(8)])
                ax.legend()
                plt.tight_layout()
                plt.savefig(str(plots_dir / "model_comparison_within.png"))
                logger.info(f"Saved model comparison plot to {plots_dir / 'model_comparison_within.png'}")
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {e}")
    
    return True


def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(description='Visualize EEG data and model results')
    parser.add_argument('--type', type=str, choices=['data', 'results', 'all'], default='all',
                        help='Type of visualization (default: all)')
    parser.add_argument('--model', type=str, choices=['sepconv1d', 'cnn1', 'all'], default='all',
                        help='Model type to visualize results for (default: all)')
    parser.add_argument('--data-path', type=str, help='Path to processed data directory')
    parser.add_argument('--results-path', type=str, help='Path to results directory')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override configuration with command-line arguments if provided
    if args.data_path:
        config.data.processed_data_path = args.data_path
    if args.results_path:
        config.data.results_path = args.results_path
    
    success = True
    
    # Perform visualizations based on type
    if args.type in ['data', 'all']:
        success = visualize_eeg_data(config) and success
    
    if args.type in ['results', 'all']:
        if args.model == 'all':
            success = visualize_model_results(config, model_type='sepconv1d') and success
            success = visualize_model_results(config, model_type='cnn1') and success
            success = visualize_model_results(config, model_type='all') and success
        else:
            success = visualize_model_results(config, model_type=args.model) and success
    
    if success:
        logger.info("Visualization completed successfully")
    else:
        logger.error("Visualization encountered errors")


if __name__ == "__main__":
    main()
