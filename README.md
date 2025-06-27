---

# NLP based Signal Detection for Project 8

This repository provides tools to simulate, preprocess, train, and validate machine learning triggering models in Python (TensorFlow 2) for weak and noisy CRES signal classification. It supports both LSTM and Transformer-based neural network architectures to learn from the chirping signature of CRES signal in Fourier space, and leverages the [Locust](https://github.com/project8/locust_mc) and [Kassiopeia](https://github.com/project8/Kassiopeia) simulation frameworks. MLFlow hyperparameter optimization with multi-GPU parallelization has been used for optimization.


---

## Project Structure

```bash
.
├── machine_learning.def                      # Singularity container definition
├── Kass_config_P8_Cavity_Template.xml        # Kassiopeia configuration template
├── LocustCavityConfig.json                   # Locust configuration template
├── GenerateCavitySims_dSQ.py                 # Simulation commands generator
├── scripts/
│   └── hist_extractor_from_root_files_mf.py  
│   └── fft_extractor_from_ts_in_root_files_mf.py  
│   └── matched_filter_algorithm.py  
├── mlflow_signal_classifier.py               # LSTM model with hyperparameter sweep
├── mlflow_signal_classifier_best_hyperparam_search_ckpt.py 
├── mlflow_locust_save_tf_trans.py            # Transformer-based model version
├── ML_Triggering_P8.ipynb                    # Jupyter notebook version
└── ...
```

---

##  Setup Instructions

### 1. Build and Run the Singularity Environment

To ensure all dependencies are consistent across systems, use the provided Singularity definition file:

```bash
singularity build ml_env.sif machine_learning.def
singularity shell ml_env.sif
```

---

## Simulation Workflow

### 2. Configure Simulation Inputs

Before running the simulation, if needed:

* Edit `Kass_config_P8_Cavity_Template.xml` for Kassiopeia settings
* Edit `LocustCavityConfig.json` for Locust parameters

These files define the cavity parameters and other experimental setups.

### 3. Generate and Run Simulations

Use the script to prepare simulations for different electron parameter configurations:

```bash
python3 GenerateCavitySims_dSQ.py
```

This script will generate command-line interface (CLI) calls to launch the Locust-Kassiopeia simulation pipeline.

---

## Dataset Preparation

Once simulations complete, convert the output ROOT files into structured datasets for ML training:

```bash
python3 scripts/hist_extractor_from_root_files_mf.py
```

This can be modified appropriately to generate datasets for training, validation, and testing.

---

## Model Training

### 4. LSTM-Based Model Training

To train models using LSTM layers with a grid of hyperparameters:

```bash
python3 mlflow_signal_classifier.py
```

To enable early stopping and eliminate low-performing hyperparameter combinations during training:

```bash
python3 mlflow_signal_classifier_best_hyperparam_search_ckpt.py
```

### 5. Transformer-Based Model

To train a Transformer-based version of the model instead of LSTM:

```bash
python3 mlflow_locust_save_tf_trans.py
```

---

## Jupyter Notebook Version

A Jupyter notebook for exploratory analysis or interactive training is available:

```bash
ML_Triggering_P8.ipynb
```

This can be used as a starting point for visual experimentation and new development.

---

## Output

Model checkpoints, MLFlow experiment tracking, and logs will be saved in the respective output directories as configured inside the scripts.
