# -*- coding: utf-8 -*-


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

#######################################################

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Conv2D, Dropout, Flatten, Reshape, BatchNormalization
#from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


############################## Load previously created dataset ######################
# load dataset 
dataframe = pd.read_csv("/home/mxm1287/timeseries_row_test_new.csv", header=None)

####### mount data from google drive ############
#from google.colab import drive
#drive.mount('/content/drive')
#dataframe = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/timeseries_row_test_new.csv", header=None)

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:1500].astype(float)
Y = dataset[:,1500]

print(Y)

def create_model(alpha = 0.25, gamma = 0.1, f1_threshold = 0.4, lstm_layers1 = 32, lstm_layers2 = 40, dropout = 0.1):
    
    model = Sequential()
    
    model.add(Reshape((10,150), input_shape = (1500,)))
    
    model.add(LSTM(lstm_layers1, activation='tanh', return_sequences=True))
    #model.add(Dropout(0))
    
    model.add(BatchNormalization())
    
    model.add(LSTM(lstm_layers2, activation='tanh'))
    
    model.add(Dropout(dropout))
    
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dropout(0.1))
    
    f1_score = tfa.metrics.F1Score(num_classes=1, threshold=f1_threshold)
    
    # Compile model 
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma = gamma), optimizer= "SGD", metrics= [f1_score])
    
    return model

# Enable MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("cres_signal_detection")

# Define a function to train and evaluate the model with MLflow tracking
def train_and_evaluate(alpha = 0.25, gamma = 0.1, f1_threshold = 0.4, lstm_layers1 = 32, lstm_layers2 = 40, dropout = 0.1):
    
    with mlflow.start_run():
        # Create model
        model = create_model(alpha, gamma, f1_threshold, lstm_layers1, lstm_layers2, dropout)
        # Train model
        plot_progress = model.fit(X, Y, validation_split=0.25, epochs=5, batch_size=10, shuffle=True, verbose=1)

        # Log parameters and metrics with MLflow
        mlflow.log_param("f1_threshold", f1_threshold)
        mlflow.log_param("num_neurons_in_first_LSTM", lstm_layers1 )
        mlflow.log_param("num_neurons_in_second_LSTM", lstm_layers2)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("f1_score", plot_progress.history['f1_score'][-1])
        mlflow.log_metric("val_f1_score", plot_progress.history['val_f1_score'][-1])
        mlflow.log_metric("loss", plot_progress.history['loss'][-1])
        mlflow.log_metric("val_loss", plot_progress.history['val_loss'][-1])
        # Save model with MLflow
        mlflow.keras.log_model(model, "model")

# Train and evaluate model with different parameters

num_neurons = [32,40]

for lstm_layers1 in num_neurons:
    for lstm_layers2 in num_neurons:
      for f1_threshold in np.arange(0.1, 0.5, 0.1):
        for alpha in np.arange(0.1, 0.5, 0.1):
          for gamma in np.arange(0.1,1.5,0.5):
                           
            train_and_evaluate(alpha, gamma, f1_threshold, lstm_layers1, lstm_layers2)



