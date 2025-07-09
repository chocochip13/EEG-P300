#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for preprocessing functions.
"""
import os
import sys
import unittest
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eeg_p300.preprocessing.preproc import data_extract, set_info, add_stim, bp_filter


class MockData:
    """Mock data class for testing preprocessing functions."""
    
    @staticmethod
    def create_mock_dataset():
        """Create a mock dataset for testing."""
        # Create mock channel names
        channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
        
        # Create mock EEG data (1000 samples, 8 channels)
        X = np.random.rand(1000, 8)
        
        # Create mock labels (0 or 1)
        y = np.random.randint(0, 2, 1000)
        
        # Create mock stimulus data
        y_stim = np.zeros(1000)
        
        # Create mock trial indices
        trial = np.arange(1000)
        
        # Create mock classes
        classes = np.array([1, 2])
        
        # Create mock classes stimulus
        classes_stim = np.array([1, 2])
        
        # Create mock subject metadata
        gender = 1
        age = 30
        ALSfrs = 10
        onsetALS = 5
        
        # Create mock data structure similar to the .mat files
        data = np.zeros((1, 1, 11), dtype=object)
        data[0, 0, 0] = np.array([[ch] for ch in channels])
        data[0, 0, 1] = X
        data[0, 0, 2] = y
        data[0, 0, 3] = y_stim
        data[0, 0, 4] = trial
        data[0, 0, 5] = classes
        data[0, 0, 6] = classes_stim
        data[0, 0, 7] = gender
        data[0, 0, 8] = age
        data[0, 0, 9] = ALSfrs
        data[0, 0, 10] = onsetALS
        
        return {'data': data}


class TestPreprocessing(unittest.TestCase):
    """Test case for preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_dataset = MockData.create_mock_dataset()
    
    def test_data_extract(self):
        """Test data extraction from dataset."""
        channels, X, y, y_stim, trial, classes, classes_stim, gender, age, ALSfrs, onsetALS = data_extract(self.mock_dataset)
        
        # Test shape and types
        self.assertEqual(len(channels), 8)
        self.assertEqual(X.shape, (1000, 8))
        self.assertEqual(y.shape, (1000,))
        self.assertEqual(y_stim.shape, (1000,))
        self.assertEqual(trial.shape, (1000,))
        self.assertEqual(classes.shape, (2,))
        self.assertEqual(classes_stim.shape, (2,))
        self.assertEqual(gender, 1)
        self.assertEqual(age, 30)
        self.assertEqual(ALSfrs, 10)
        self.assertEqual(onsetALS, 5)
    
    def test_set_info(self):
        """Test creating MNE info object."""
        channels, X, *_ = data_extract(self.mock_dataset)
        raw = set_info(X, channels)
        
        # Test raw object properties
        self.assertEqual(raw.info['nchan'], 8)
        self.assertEqual(raw.info['sfreq'], 256)
    
    def test_add_stim(self):
        """Test adding stimulus channel to raw data."""
        channels, X, y, *_ = data_extract(self.mock_dataset)
        raw = set_info(X, channels)
        raw_with_stim = add_stim(raw, y)
        
        # Test raw object with stim channel
        self.assertEqual(raw_with_stim.info['nchan'], 9)  # 8 EEG + 1 stim
        self.assertIn('STI', raw_with_stim.ch_names)
    
    def test_bp_filter(self):
        """Test bandpass filtering."""
        # This is more of an integration test
        filt_raw = bp_filter(self.mock_dataset)
        
        # Test filtered raw object
        self.assertEqual(filt_raw.info['nchan'], 9)  # 8 EEG + 1 stim
        self.assertIn('STI', filt_raw.ch_names)


if __name__ == '__main__':
    unittest.main()
