# -*- coding: utf-8 -*-
'''
Author: R. Mohiuddin
Uses MLFlow to log parameters and return the f1_scores
Parallelized version for faster processing following https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map

'''


import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import minmax_scale
import scipy
from scipy.fft import rfft, rfftfreq
import pandas as pd
import random
from datetime import datetime
random.seed(datetime.now().timestamp())
import mlflow
import mlflow.keras
import mlflow.tensorflow
import multiprocessing as mp
import time
import os



#######################################################

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Conv2D, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


############################## Load previously created dataset ######################
# load training dataset 
dataframe = pd.read_csv("timeseries_row_test_new.csv", header=None)

####### mount data from google drive ############
#from google.colab import drive
#drive.mount('/content/drive')
#dataframe = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/timeseries_row_test_new.csv", header=None)

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:1500].astype(float)
Y = dataset[:,1500]

############# For Validation ###################
#load validation dataset
val_dataframe = pd.read_csv("timeseries_row_validation.csv", header=None)
val_dataset = val_dataframe.values
# split into input (X_val) and output (Y_val) variables
X_val = val_dataset[:,0:1500].astype(float)
Y_val = val_dataset[:,1500]

print(Y)

    
def create_model(alpha = 0.25, gamma = 0.1, f1_threshold = 0.4, lstm_layers1 = 32, dense_neurons = 32, lstm_layers2 = 40, dropout = 0, learning_rate = 0.005):
    
    model = Sequential()
    
    model.add(Reshape((100,15), input_shape = (1500,)))
    
    model.add(LSTM(lstm_layers1, activation='tanh', return_sequences=True))
    #model.add(Dropout(0))
    
    model.add(BatchNormalization())
    
    model.add(LSTM(lstm_layers2, activation='tanh'))
    
    model.add(Dropout(dropout))
    
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dropout(0.1))
    
    f1_score = tfa.metrics.F1Score(num_classes=1, threshold=f1_threshold)
    
    SGD = tf.keras.optimizers.SGD(learning_rate = learning_rate)
    
    # Compile model 
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma = gamma), optimizer= SGD, metrics= [f1_score])
    
    return model

# callback for quick save

checkpoint_filepath = '/home/mxm1287/ML/mlruns'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_f1_score',
    mode='max',
    save_best_only=False)



# Enable MLflow tracking

mlflow.set_tracking_uri("http://localhost:5000")

# Enable auto-logging to MLflow to capture TensorBoard metrics.
#mlflow.tensorflow.autolog()
mlflow.set_experiment("cres_signal_detection_no_clbck_1")


# Define hyperparameters to search over
hyperparameters = {'alpha': [ 0.75, 0.2],
                   'gamma': [0.1, 2],
                   'f1_threshold': [0.4, 0.8],
                   'lstm_layers1': [64, 32],
                   'lstm_layers2': [64],
                   'dropout': [0.0],
                   'learning_rate': [0.001]}

# Generate a list of hyperparameter combinations to try
param_grid = list(ParameterGrid(hyperparameters))
#print(param_grid)

# Shuffle the hyperparameter combinations for randomness
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
    # Generate a list of models to try for this epoch
    models_to_try = []
    for params in param_grid:
     #   print(params)
        model = create_model(**params)
        models_to_try.append(model)

    
    # Train each model for one epoch and save the f1 scores
    f1_scores = []

    print(f"Num of models being evaluated: {len(models_to_try)}")
    #print(len(models_to_try))

    for model in models_to_try:
        history = model.fit(X, Y, epochs=1, validation_data=(X_val, Y_val), callbacks = [model_checkpoint_callback], use_multiprocessing = True, verbose=1)
        f1_scores.append(history.history['val_f1_score'][-1])

        # Log the training progress to MLflow
        with mlflow.start_run():

            mlflow.keras.log_model(model, "model")
            
    
    # Find the indices of the best and worst performing hyperparameters
    best_indices = np.argsort(f1_scores)[-num_to_keep:]
    worst_indices = np.argsort(f1_scores)[:num_to_discard]
    
    # Print the best and worst performing hyperparameters for this epoch
   # print(f"Best hyperparameters for epoch {epoch+1}:")
    #for i in best_indices:
     #   print(param_grid[i])
    print(np.argsort(f1_scores))
    print(worst_indices)
#    print(f"Worst hyperparameters for epoch {epoch+1}:")
#    for i in worst_indices:
#        print(param_grid[i])
    
    # Discard the worst performing hyperparameters
    param_grid = [param_grid[i] for i in range(len(param_grid)) if i not in worst_indices]
    print(f"Last tested parameters: {param_grid}")
    
    # Reduce the number of hyperparameters to try for the next epoch
    num_to_try = len(param_grid)
    if num_to_try < num_to_keep:
        break  # Stop searching if we have found enough good hyperparameters



