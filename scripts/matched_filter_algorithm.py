import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_signal(template, noise_level=0.02):
    noise = noise_level * np.random.randn(len(template))
    return template + noise

def matched_filter(signal, template):

    # Time-reverse and conjugate the template
    template_reversed = template[::-1]
    template_conjugated = np.conj(template_reversed)

    # Convolve the signal with the conjugated, time-reversed template
    correlation = np.convolve(signal, template_conjugated, mode='valid')
#   correlation = np.correlate(signal, template, mode='valid')
    return abs(correlation)
"""

def matched_filter(signal, template, chunk_length=8192, overlap=0):
    # Time-reverse and conjugate the template
    template_reversed = template[::-1]
    template_conjugated = np.conj(template_reversed)

    # Overlap-save implementation:
    output_buffer = np.zeros(len(signal) + len(template) - 1)
    chunk_offset = 0
    while chunk_offset + chunk_length < len(signal):
        chunk = signal[chunk_offset:chunk_offset + chunk_length]
        filtered_chunk = np.convolve(chunk, template_conjugated, mode='valid')
        output_buffer[chunk_offset + M - 1:chunk_offset + chunk_length + M - 1] += filtered_chunk
        chunk_offset += chunk_length - overlap

    return output_buffer """

def detect_event(correlation, threshold):
    peaks = np.where(correlation > threshold)[0]
    if peaks.size > 0:
        print(correlation, peaks)
        return max(correlation)
    return 0

#Loading all templates (signal-only data)
dataframe = pd.read_csv("/gpfs/gibbs/project/heeger/rm2498/trig_ml/8675_degree_electron_tracks_20_80_mf.csv", header=None)
template_bank = dataframe.to_numpy()

signal = []
# Loading noisy signal data to apply template (same signal data with added gaussian noise)
for sig in template_bank:
    signal.append(generate_signal(sig))


#correlation_result = matched_filter(signal, template)
events = []
threshold = 0.33 #0.526

for signal_slice in signal:
    
    max_correlation = 0
    for template in template_bank:
        corr = detect_event(matched_filter(signal_slice, template), threshold)
        if corr > max_correlation:
            max_correlation = corr
    try:
        events.append(max_correlation)
    except:
        print("pass")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(signal[19], label='Signal with Noise')
plt.plot(template_bank[19], label='Signal', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Matched Filter Example')
plt.legend()
plt.grid(which = 'both', linestyle = "dashed")
#plt.legend(loc='lower right')
plt.savefig(f'plot_matched_filter')

print("Detected events:", events)






###################    ROC      #########################










from tensorflow import keras

# plotting ROC curve for the test set
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

def optimal_thresh_from_roc(target, predicted):
    """
    Find optimal threshold for classification from roc curve
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "threshold": pd.Series(threshold, index=i),
        }
    )
    roc_t = roc.iloc[
        (roc.tf - 0).abs().argsort()[:1]
    ]  # optimal value is where tpr is high and fpr is low
    # i.e. tpr - (1-fpr) is/is near zero

    return roc_t["threshold"].values[0]

def plot_roc_and_confusion(target, predicted, label="Matched Filter"):
    """
    Plot roc curve + auc and confusion matrix for specified predictions
    """

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(target, predicted)
    plt.figure(figsize=(8, 6))
    plt.grid()
    plt.plot(fpr, tpr, label=f"{label}", color="darkblue")
    plt.plot([0, 1], [0, 1], linestyle="dashed", label="random", color="gray")
    plt.legend(loc="lower right")
    plt.ylabel("TPR", fontsize=14)
    plt.xlabel("FPR", fontsize=14)
    plt.title(
        f"{label} (varying survival threshold), AUC = {auc(fpr,tpr):.5f}",
        fontsize=14,
    )
    #plt.show()
    plt.savefig(f'mf_rec_op_curve_0_02')
    plt.clf()

    # Confusion Matrix
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    optimal_thresh = optimal_thresh_from_roc(target, predicted)
    alive_pred_lr_optimal = np.where(predicted >= optimal_thresh, 1, 0)
    cm = confusion_matrix(target, alive_pred_lr_optimal)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    ax.set_title(
        f"{label} Survival in Holdout"
        + "\n"
        + f"Optimal Thresh = {optimal_thresh:.2f}"
    )
    #plt.show()
    plt.savefig(f'mf_conf_matrix_0_02')
    plt.clf()


# Loading labels for testing
label_df = pd.read_csv("//gpfs/gibbs/project/heeger/rm2498/trig_ml/label_8675_degree_electron_tracks_20_80_mf.csv", header=None) #, usecols = [i for i in range(8192)], nrows= 100)
Y_test = label_df.to_numpy()
print(Y_test)


plot_roc_and_confusion(Y_test, events)



