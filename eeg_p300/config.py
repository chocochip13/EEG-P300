"""
Configuration management for EEG-P300 project.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import os

@dataclass
class DataConfig:
    """Configuration for data paths and settings."""
    raw_data_path: str = os.environ.get("EEG_RAW_DATA_PATH", "data/raw")
    processed_data_path: str = os.environ.get("EEG_PROCESSED_DATA_PATH", "data/processed")
    results_path: str = os.environ.get("EEG_RESULTS_PATH", "data/results")

@dataclass
class PreprocessingConfig:
    """Configuration for EEG signal preprocessing."""
    f_low: float = 0.1
    f_high: float = 30
    tmin: float = -0.2
    tmax: float = 0.8
    sfreq: int = 256
    iir_params: Dict = None

    def __post_init__(self):
        if self.iir_params is None:
            self.iir_params = dict(order=8, ftype='butter')

@dataclass
class ModelConfig:
    """Configuration for model training and architecture."""
    model_type: str = "sepconv1d"  # or "cnn1"
    filters: int = 32
    batch_size: int = 256
    epochs: int = 200
    patience: int = 50
    random_seed: int = 42

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    cross_validation_folds: int = 5
    cross_validation_repeats: int = 10
    test_size: float = 0.2
    random_seed: int = 42

@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

# Default configuration instance
config = Config()
