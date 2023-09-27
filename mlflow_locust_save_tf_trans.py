# -*- coding: utf-8 -*-
'''
Author: R. Mohiuddin
Uses MLFlow to log parameters and return the f1_scores, saves tf model in a seperate folders
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Flatten, Reshape, BatchNormalization, MultiHeadAttention, LayerNormalization
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import Callback


############################## Load previously created dataset ######################
# load training dataset 
dataframe = pd.read_csv("timeseries_locust_test.csv", header=None)

####### mount data from google drive ############
#from google.colab import drive
#drive.mount('/content/drive')
#dataframe = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/timeseries_row_test_new.csv", header=None)

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:8192].astype(float)
Y = dataset[:,8192]

############# Stratified Shuffle Split for Validation ###################

# Create an instance of StratifiedShuffleSplit
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=420)

# X_val and Y_val are validation features and labels
for train_index, val_index in stratified_splitter.split(X, Y):
    X, X_val = X[train_index], X[val_index]
    Y, Y_val = Y[train_index], Y[val_index]

# Now, new X and Y are training data, and X_val and Y_val are stratified validation data.


print(Y)

def create_model(alpha = 0.25, gamma = 0.1, f1_threshold = 0.4, lstm_layers1 = 32, dense_neurons = 32, lstm_layers2 = 8, dropout = 0):
    
        
    # Input data shape is (8192,)
    input_shape = (8192,)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Reshape layer (optional)
    reshaped = Reshape(target_shape=(512, 16))(inputs)
    
    # Transformer Encoder Block
    attention1 = MultiHeadAttention(num_heads = lstm_layers1, key_dim=lstm_layers2)(reshaped, reshaped, reshaped)
    norm1_1 = LayerNormalization(epsilon=1e-6)(attention1)
    ffn1 = tf.keras.layers.Dense(dense_neurons, activation='relu')(norm1_1)
    norm2_1 = LayerNormalization(epsilon=1e-6)(ffn1)
    
    # Replacing LSTM layers with Transformer Encoder layers
    transformer_output1 = LayerNormalization(epsilon=1e-6)(norm2_1)
    
    
     # More Transformer Encoder Block
    attention2 = MultiHeadAttention(num_heads = lstm_layers1, key_dim=lstm_layers2)(transformer_output1, transformer_output1, transformer_output1)
    norm1_2 = LayerNormalization(epsilon=1e-3)(attention2)
    ffn_2 = tf.keras.layers.Dense(dense_neurons, activation='relu')(norm1_2)
    norm2_2 = LayerNormalization(epsilon=1e-3)(ffn_2)
    
    # Replacing LSTM layers with Transformer Encoder layers 
    transformer_output = LayerNormalization(epsilon=1e-3)(norm2_2)
    
    
    
    # Flatten
    flatten = Flatten()(transformer_output)
    
    # Dense layers
    dense1 = Dense(dense_neurons, activation='relu')(flatten)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(dense1)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # F1 Score metric
    f1_score = tfa.metrics.F1Score(num_classes=1, threshold=f1_threshold)
    
    
    SGD = tf.keras.optimizers.SGD(learning_rate = 0.001)
    
    # Compile model 
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma = gamma), optimizer= SGD, metrics= [f1_score, Recall(), Precision()])
    
    return model

# Enable MLflow tracking

mlflow.set_tracking_uri("http://localhost:5000")

# Enable auto-logging to MLflow to capture TensorBoard metrics.
#mlflow.tensorflow.autolog()
mlflow.set_experiment("locust_save_tf_trans_2")


# Define a callback to select the best hyperparameters after each epoch
class HyperparameterCallback(Callback):        
    """Custom Keras callback that selects the best hyperparameters based on validation loss"""
    def __init__(self, params_list):
        super(HyperparameterCallback, self).__init__()
        self.params_list = params_list
        self.best_params = []
        self.best_f1_score = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_f1_score'] < self.best_f1_score:
            self.best_f1_score = logs['val_f1_score']
            self.best_params = self.params_list[0]
        self.params_list = self.params_list[1:]
        if len(self.params_list) == 0:
            self.model.stop_training = False


num_neurons = [16, 32]
num_dense_neurons = [16, 32]
gammas = [0, 0.1]
alphas = [0.35, 0.8, 0.1]
f1_thresholds = [0.09, 0.2]
dropouts = [0, 0.1, 0.2]




# Define a function to train and evaluate the model with MLflow tracking
def train_and_evaluate(alpha = 0.25, gamma = 0.1, f1_threshold = 0.4, lstm_layers1 = 32, dense_neurons = 32, lstm_layers2 = 40, epochs=16, dropout = 0):
    
    for dropout in dropouts:
        for lstm_layers1 in num_neurons:
            for dense_neurons in num_dense_neurons:
                for lstm_layers2 in num_neurons:
                    for f1_threshold in f1_thresholds:
                        for alpha in alphas:
                            for gamma in gammas:
                                model_dir = f"tf_models/{dropout}{lstm_layers1}{lstm_layers2}{f1_threshold}{alpha}{gamma}{dense_neurons}"
                                model_dir = model_dir.replace(".", "_")
                                model_dir = model_dir.replace("[", "")
                                model_dir = model_dir.replace("]", "")
                               # os.path.join(".", model_dir)
                                
                                
                                # defining callback
                                callback = HyperparameterCallback(params_list=[{"f1_threshold": f1_threshold,
                                                        "num_neurons_in_first_LSTM": lstm_layers1,
                                                        "num_Dense_neurons": dense_neurons,
                                                        "num_neurons_in_second_LSTM": lstm_layers2,
                                                        "dropout": dropout,
                                                        "gamma": gamma,
                                                        "alpha": alpha,
                                                         }])
                                
                                try:
                                    # loading the saved model
                                    loaded_model = tf.keras.models.load_model(model_dir)
                                    
                                    print(f"Already trained model found, resuming training from {model_dir}")
    
                                    
                                    # Display model summary
                                    loaded_model.summary()
                                    # resuming model from previous checkpoint
                                    plot_progress = loaded_model.fit(X, Y, 
                                                        validation_split=0.3, 
                                                        #validation_data=(X_val, Y_val),
                                                        epochs= epochs, 
                                                        batch_size=100, 
                                                        shuffle=True, 
                                                        callbacks=[callback], 
                                                        verbose=1)
    
                                except:
    #                                continue
                                    os.makedirs(model_dir)
                                    
                                    # saving the model in tensorflow format
                                    print(f'No model exists for this set of hyperparameters, saving model at {model_dir}')
                                    
                                    
                                    # Create model
                                    model = create_model(alpha, gamma, f1_threshold, lstm_layers1, dense_neurons, lstm_layers2, dropout)
    
                                   
                                    # Display model summary
                                    model.summary()
                                    
                                    # Training model
                                    plot_progress = model.fit(X, Y, 
                                                        validation_split=0.3, 
                                                        #validation_data=(X_val, Y_val),
                                                        epochs= epochs, 
                                                        batch_size=100, 
                                                        shuffle=True,
                                                        callbacks=[callback], 
                                                        verbose=1)
    
                                    model.save(model_dir, save_format='tf')
    
                                
                                
                                # Log the training progress to MLflow
                                with mlflow.start_run():
                                    # Log parameters and metrics with MLflow
                                    mlflow.log_param("f1_threshold", f1_threshold)
                                    mlflow.log_param("first_trans_heads", lstm_layers1 )
                                    mlflow.log_param("num_Dense_neurons", dense_neurons )
                                    mlflow.log_param("second_trans_heads", lstm_layers2)
                                    mlflow.log_param("dropout", dropout)
                                    mlflow.log_param("gamma", gamma)
                                    mlflow.log_param("alpha", alpha)
            
                                    mlflow.log_metric("f1_score", np.max(plot_progress.history['f1_score']))
                                    mlflow.log_metric("val_f1_score", np.max(plot_progress.history['val_f1_score']))
                                    mlflow.log_metric("loss", plot_progress.history['loss'][-1])
                                    mlflow.log_metric("val_loss", plot_progress.history['val_loss'][-1])
    
    return np.max(plot_progress.history['f1_score'])

# Run the hyperparameter search and log the results to MLflow

if __name__ == '__main__':
    train_and_evaluate()
