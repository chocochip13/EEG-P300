#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to prepare EEG P300 data for analysis and modeling.
"""
import os
import numpy as np
import scipy.io
import glob
import logging
import sys
import argparse
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eeg_p300.preprocessing.preproc import bp_filter, epochs_gen
from eeg_p300.config import Config, DataConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_directories(config):
    """
    Create necessary directories for data processing.
    
    Parameters
    ----------
    config : Config
        Configuration object
    """
    # Create raw data directory if it doesn't exist
    raw_dir = Path(config.data.raw_data_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Raw data directory: {raw_dir}")
    
    # Create processed data directory if it doesn't exist
    processed_dir = Path(config.data.processed_data_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processed data directory: {processed_dir}")
    
    # Create results directory if it doesn't exist
    results_dir = Path(config.data.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir.joinpath('within_subject').mkdir(parents=True, exist_ok=True)
    results_dir.joinpath('cross_subject').mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {results_dir}")


def process_data(raw_data_path, processed_data_path):
    """
    Process raw EEG data and save processed epochs.
    
    Parameters
    ----------
    raw_data_path : str
        Path to raw data directory
    processed_data_path : str
        Path to processed data directory
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Find all .mat files in the raw data directory
    filepath = Path(raw_data_path)
    df = list(filepath.glob('*.mat'))
    
    if not df:
        logger.error(f"No .mat files found in {raw_data_path}")
        return False
    
    logger.info(f"Found {len(df)} .mat files")
    
    # Load all datasets
    datasets = []
    for i, file_path in enumerate(df):
        logger.info(f"Loading dataset {i+1}/{len(df)}: {file_path.name}")
        datasets.append(scipy.io.loadmat(str(file_path)))
    
    # Apply bandpass filtering and generate epochs
    filt_raw = []
    ep = []
    ev = []
    
    for i, dataset in enumerate(datasets):
        logger.info(f"Processing dataset {i+1}/{len(datasets)}")
        filt_raw.append(bp_filter(dataset))
        ep.append(epochs_gen(filt_raw[i]))
        ev.append(ep[i].average(by_event_type=True))
        
        # Save epochs in .fif format
        output_path = Path(processed_data_path) / f"A0{i+1}-epo.fif"
        ep[i].save(str(output_path))
        logger.info(f"Saved epochs to {output_path}")
    
    # Extract labels and data
    labels = []
    data = []
    for i in range(len(ep)):
        labels.append(ep[i].events[:, -1])
        data.append(ep[i].get_data())
    
    # Combine epochs and labels
    combined_epochs = np.stack(data, axis=0)
    combined_labels = np.stack(labels, axis=0)
    combined_labels = combined_labels - 1  # Adjust labels to be 0-based
    
    # Save as NumPy arrays
    np.save(Path(processed_data_path) / "A-epo.npy", combined_epochs)
    np.save(Path(processed_data_path) / "A-labels.npy", combined_labels)
    
    logger.info(f"Saved combined epochs to {processed_data_path}/A-epo.npy")
    logger.info(f"Saved combined labels to {processed_data_path}/A-labels.npy")
    
    return True


def main():
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(description='Prepare EEG P300 data')
    parser.add_argument('--raw-path', type=str, help='Path to raw data directory')
    parser.add_argument('--processed-path', type=str, help='Path to processed data directory')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override configuration with command-line arguments if provided
    if args.raw_path:
        config.data.raw_data_path = args.raw_path
    if args.processed_path:
        config.data.processed_data_path = args.processed_path
    
    # Set up directories
    setup_directories(config)
    
    # Process data
    success = process_data(config.data.raw_data_path, config.data.processed_data_path)
    
    if success:
        logger.info("Data preparation completed successfully")
    else:
        logger.error("Data preparation failed")


if __name__ == "__main__":
    main()
