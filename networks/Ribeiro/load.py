import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from scipy import optimize

STEP = 5000

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

def import_key_data(path):
    labels=[]
    ecg_filenames=[]
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                labels.append(header_data[15][5:-1])
                ecg_filenames.append(filepath)
    return labels, ecg_filenames

def load_ecg_data(ecg_filenames):
  for i in range(ecg_filenames.shape[0]):
    data, header_data = load_challenge_data(ecg_filenames[i])
    data = pad_sequences(data, maxlen=STEP, truncating='post', padding="post")
    yield data.T

def random_mix_12lead(signal):
  """
  SSL Approach 1: Mixing the channels of ECG
  """
  order = np.arange(12)
  np.random.shuffle(order)
  return signal[:, order]

def split_join_12lead(signal, no_split=2):

  new_signal = np.copy(signal)
  order = np.arange(12)
  np.random.shuffle(order)
  # pick how many channels to split and join
  no_channels = np.random.randint(0, 12, size=1)[0] 
  for i in order[0:no_channels]:
    new_signal[:,i] = np.hstack(np.split(new_signal[:,i], no_split)[::-1])
  return new_signal

def SSL_batch_generator( signal, batch_size=3):
  for i in range(ecg_filenames.shape[0]):
    data, header_data = load_challenge_data(ecg_filenames[i])
    data = pad_sequences(data, maxlen=STEP, truncating='post', padding="post")

    batch_signal_1 = data.T
    batch_signal_2 = random_mix_12lead(batch_signal_1)
    batch_signal_3 = split_join_12lead(batch_signal_1, no_split=2)
    yield np.stack((batch_1,batch_2,batch_3), axis = 0)





