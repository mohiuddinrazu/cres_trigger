import os
import uproot
import numpy as np
import pandas as pd


# function to add noise to a given array
def generate_signal(template, noise_level=0.02):
    noise = noise_level * np.random.randn(len(template))
    return template# + noise

# function to collect timeseries data from a single window and adding noise to the signal
def hist_collector(root_file):
    #try:
    data = []
    for i in range(0, 48):
        hist_TS = root_file[f'histTSReal_{i}_0;1'].to_numpy()
        timeseries = generate_signal(hist_TS[0])
        data.append(timeseries)
    return data
    
    #except Exception as e:
     #   print(e)
      #  return []
# function to collect all zeros from a single window (to generate noise only data)        
def noise_collector(root_file):
    #try:
    data = []
    for i in range(100, 295):
        hist_TS = root_file[f'histTSReal_{i}_0;1'].to_numpy()
        timeseries = generate_signal(hist_TS[0])
        data.append(timeseries)
    return data
    
    #except Exception as e:
     #   print(e)
      #  return []

def main():
    path_to_root_files = '/gpfs/gibbs/project/heeger/rm2498/data/Simulation/Cavity/root_files_by_param/'
    root_file_names = os.listdir(path_to_root_files) # grab all files from above directory

    checkpoint = 0 # For checking progress after certain number of operations

    
    # collecting signal data
    for root_file_name in root_file_names:
        file_path = path_to_root_files + root_file_name
        root_file = uproot.open(file_path)
        print(f"Working on file {file_path}")
        checkpoint += 1 # checkpoint

        data = hist_collector(root_file)
        if data:
            save_data_to_csv(np.array(data))
            save_label_to_csv(np.array([1 for i in range(0, 48)])) # labeling all windows with 1's
            
            if checkpoint%5 == 0: print(data) # Print current root file info with label
   
    # add noise only data to the end
    for root_file_name in root_file_names:
        file_path = path_to_root_files + root_file_name
        root_file = uproot.open(file_path)
        print(f"Working on file {file_path}")
        checkpoint += 1 # checkpoint

        data = noise_collector(root_file)
        if data:
            save_data_to_csv(np.array(data))
            save_label_to_csv(np.array([0 for i in range(100, 295, 1)])) # labeling all windows with 0's
            
            if checkpoint%5 == 0: print(data) # Print current root file info with label            

# function for saving data every step (step = root file). Make sure file_name already doesn't exist
def save_data_to_csv(data, file_name="8675_degree_electron_tracks_20_80_mf.csv"):
    data = np.array(data)
    df = pd.DataFrame(data)
    df.to_csv(file_name, header=None, index=None, mode="a")

# function to write true/false label in a seprate file. Make sure file_name already doesn't exist
def save_label_to_csv(data, file_name="label_8675_degree_electron_tracks_20_80_mf.csv"):
    data = np.array(data)
    df = pd.DataFrame(data)
    df.to_csv(file_name, header=None, index=None, mode="a")

if __name__ == "__main__":
    main()

