## EEG-P300: P300 Signal Detection from EEG Data

This project contains code for processing EEG signals data and prediction of P300 signals. The aim of this project is to:

* Preprocess raw EEG signals data
* Perform Exploratory Data Analysis (EDA)
* Build and evaluate prediction models for P300 detection

### Dataset

The EEG signals dataset is the P300 speller with ALS patients from [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets) datasets. 
This dataset contains 8 subjects diagnosed with ALS, with 8 channels and sampling frequency of 256Hz.
More details about the dataset can be found [here](https://lampx.tugraz.at/~bci/database/008-2014/description.pdf).

### Project Structure

```
EEG-P300/
├── data/                      # Data directory (created at runtime)
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data
│   └── results/               # Model results and visualizations
├── eeg_p300/                  # Main package directory
│   ├── config.py              # Configuration management
│   ├── preprocessing/         # Preprocessing modules
│   │   └── preproc.py         # Preprocessing functions
│   ├── models/                # Model definitions
│   │   ├── cnn1.py            # CNN1 model implementation
│   │   └── sepconv1d.py       # SepConv1D model implementation
│   ├── evaluation/            # Evaluation scripts
│   │   ├── cross_subject.py   # Cross-subject evaluation
│   │   └── within_subject.py  # Within-subject evaluation
│   └── utils/                 # Utility functions
│       ├── data_utils.py      # Data handling utilities
│       ├── metrics.py         # Evaluation metrics
│       └── visualization.py   # Plotting functions
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── CNN1.ipynb             # CNN1 model exploration
│   └── sepconv1d.ipynb        # SepConv1D model exploration
├── scripts/                   # Command-line scripts
│   ├── prepare_data.py        # Data preparation script
│   ├── train_model.py         # Model training script
│   └── visualize_results.py   # Results visualization script
└── tests/                     # Unit tests
```

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/chocochip13/EEG-P300.git
   cd EEG-P300
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Activate the virtual environment:
   ```
   poetry shell
   ```

### Usage

#### Data Preparation

1. Download the P300 speller dataset from [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets) and place the .mat files in the `data/raw` directory.

2. Run the data preparation script:
   ```
   python scripts/prepare_data.py
   ```

   Options:
   - `--raw-path`: Path to raw data directory (default: data/raw)
   - `--processed-path`: Path to processed data directory (default: data/processed)

#### Model Training

Train models using the following script:
```
python scripts/train_model.py
```

Options:
- `--model`: Model type to use (sepconv1d or cnn1, default: sepconv1d)
- `--eval-type`: Evaluation strategy (within or cross, default: within)
- `--data-path`: Path to processed data directory
- `--results-path`: Path to results directory
- `--batch-size`: Batch size for training
- `--epochs`: Maximum number of epochs
- `--filters`: Number of filters (for sepconv1d)
- `--patience`: Patience for early stopping

Examples:
```
# Train SepConv1D with within-subject evaluation
python scripts/train_model.py --model sepconv1d --eval-type within

# Train CNN1 with cross-subject evaluation
python scripts/train_model.py --model cnn1 --eval-type cross
```

#### Visualization

Visualize data and results:
```
python scripts/visualize_results.py
```

Options:
- `--type`: Type of visualization (data, results, or all, default: all)
- `--model`: Model type to visualize results for (sepconv1d, cnn1, or all, default: all)
- `--data-path`: Path to processed data directory
- `--results-path`: Path to results directory

Examples:
```
# Visualize all data and results
python scripts/visualize_results.py

# Visualize only EEG data
python scripts/visualize_results.py --type data

# Visualize only SepConv1D results
python scripts/visualize_results.py --type results --model sepconv1d
```

### Models

#### CNN1

CNN1 is a 1D convolutional neural network for EEG signal classification based on [Cecotti et al. 2011](https://ieeexplore.ieee.org/document/5492691). It uses:
- Custom weight initialization (cecotti_normal)
- Scaled tanh activation function
- Two 1D convolutional layers
- Two dense layers with sigmoid activation

#### SepConv1D

SepConv1D is a simpler model using separable convolutions:
- Zero padding
- One separable 1D convolution layer
- Tanh activation
- Dense layer with sigmoid activation

### Evaluation Strategies

#### Within-Subject Evaluation

- Trains and evaluates models for each subject separately
- Uses Repeated Stratified K-fold cross-validation
- Results are averaged across folds and repeats

#### Cross-Subject Evaluation

- Uses Leave-One-Subject-Out cross-validation
- Trains on data from all subjects except one
- Tests on the left-out subject
- Results show generalization across subjects

### Results

Results are saved in the `data/results` directory, with separate subdirectories for `within_subject` and `cross_subject` evaluations. Visualizations are saved in the `data/results/plots` directory.

### Credits

This project is maintained by [chocochip13](https://github.com/chocochip13).
