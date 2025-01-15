# -*- coding: utf-8 -*-
"""
Author: R. Mohiuddin
Uses MLFlow to log parameters and return the f1_scores
Parallelized version for faster processing following https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map
"""

#Import database manipulation modules
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import minmax_scale
import scipy
from scipy.fft import rfft, rfftfreq
import pandas as pd

#Import system modules
import multiprocessing as mp
import time
import os
import random
from datetime import datetime

#Import TF related modules
import mlflow
import mlflow.keras
import mlflow.tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import cross_val_score, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

random.seed(datetime.now().timestamp())

# Load previously created dataset
dataframe = pd.read_csv("timeseries_row_test_new.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:1500].astype(float)
Y = dataset[:, 1500]

# Load validation dataset
val_dataframe = pd.read_csv("timeseries_row_validation.csv", header=None)
val_dataset = val_dataframe.values
X_val = val_dataset[:, 0:1500].astype(float)
Y_val = val_dataset[:, 1500]

print(Y)

def create_model(alpha=0.25, gamma=0.1, f1_threshold=0.4, lstm_layers1=32, dense_neurons=32, lstm_layers2=40, dropout=0, learning_rate=0.005):
    """
    Create and compile the LSTM model.
    
    Parameters
    ----------
    alpha : float, optional
        Alpha parameter for BinaryFocalCrossentropy, by default 0.25.
    gamma : float, optional
        Gamma parameter for BinaryFocalCrossentropy, by default 0.1.
    f1_threshold : float, optional
        Threshold for F1 Score, by default 0.4.
    lstm_layers1 : int, optional
        Number of units in the first LSTM layer, by default 32.
    dense_neurons : int, optional
        Number of neurons in the Dense layer, by default 32.
    lstm_layers2 : int, optional
        Number of units in the second LSTM layer, by default 40.
    dropout : float, optional
        Dropout rate, by default 0.
    learning_rate : float, optional
        Learning rate for the SGD optimizer, by default 0.005.
    
    Returns
    -------
    Sequential
        Compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Reshape((100, 15), input_shape=(1500,)))
    model.add(LSTM(lstm_layers1, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(lstm_layers2, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    f1_score = tfa.metrics.F1Score(num_classes=1, threshold=f1_threshold)
    SGD = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma=gamma), optimizer=SGD, metrics=[f1_score])
    return model

checkpoint_filepath = '/home/mxm1287/ML/mlruns'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_f1_score',
    mode='max',
    save_best_only=False
)

# Enable MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("cres_signal_detection_no_clbck_1")

# Define hyperparameters to search over
hyperparameters = {
    'alpha': [0.75, 0.2],
    'gamma': [0.1, 2],
    'f1_threshold': [0.4, 0.8],
    'lstm_layers1': [64, 32],
    'lstm_layers2': [64],
    'dropout': [0.0],
    'learning_rate': [0.001]
}

# Generate a list of hyperparameter combinations to try
param_grid = list(ParameterGrid(hyperparameters))
np.random.shuffle(param_grid)

# Set number of epochs to run
num_epochs = 15

# Set number of worst performing hyperparameters to discard after each epoch
num_to_discard = 15

# Set number of best performing hyperparameters to keep at the end
num_to_keep = 10

# Set initial number of hyperparameters to try
num_to_try = len(param_grid)

# Run the model for multiple epochs and hyperparameters
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    models_to_try = [create_model(**params) for params in param_grid]
    f1_scores = []
    print(f"Num of models being evaluated: {len(models_to_try)}")
    for model in models_to_try:
        history = model.fit(X, Y, epochs=1, validation_data=(X_val, Y_val), callbacks=[model_checkpoint_callback], use_multiprocessing=True, verbose=1)
        f1_scores.append(history.history['val_f1_score'][-1])
        with mlflow.start_run():
            mlflow.keras.log_model(model, "model")
    
    best_indices = np.argsort(f1_scores)[-num_to_keep:]
    worst_indices = np.argsort(f1_scores)[:num_to_discard]
    print(np.argsort(f1_scores))
    print(worst_indices)
    param_grid = [param_grid[i] for i in range(len(param_grid)) if i not in worst_indices]
    print(f"Last tested parameters: {param_grid}")
    num_to_try = len(param_grid)
    if num_to_try < num_to_keep:
        break
