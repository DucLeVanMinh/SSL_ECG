import json
import tqdm
import scipy.io as sio
import numpy as np
import random
from tensorflow import keras

random.seed(86)

STEP = 256

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    ecg = sio.loadmat(record)['val'].squeeze()
    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

def split_join_1lead(signal, no_split=2):
  return np.hstack(np.split(signal, no_split)[::-1])

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

class Preproc:
    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32) 
        y = keras.utils.to_categorical(
                y, num_classes=len(self.classes))
        return y

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    # random.shuffle(batches)
    while True:
        random.shuffle(batches)
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

def SSL_generator(signal):
  mean, std = compute_mean_std(signal)
  num_examples = len(signal)
  examples = sorted(signal, key = lambda x: x.shape[0])
  # random.shuffle(examples)
  while True:
    random.shuffle(examples)
    for sig in examples:
      origin_sig = (sig - mean)/std
      ssl_sig    = (split_join_1lead(sig) - mean)/std
      ssl_sig_2  = (split_join_1lead(sig, 4) - mean)/std
      ssl_sig_3  = (split_join_1lead(sig, 8) - mean)/std
      # batch = np.stack((origin_sig[:,None], 
      #                   ssl_sig[:,None],
      #                   ssl_sig_2[:,None],
      #                   ssl_sig_3[:,None]), axis=0)
      batch = [origin_sig,
               ssl_sig,
               ssl_sig_2]
              #  ssl_sig_3]
      yield batch

def SSL_batch_generator(batch_size, data_gen, data_size):
  while True:
    for i in range(int(data_size/batch_size)):
      batch = []
      for i in range(batch_size):
        batch.append(next(data_gen)) 

      random.shuffle(batch)
      batches = []
      for i in range(batch_size):
        batches += batch[i]

      yield SSL_process(batches)

def SSL_process(x):
  x = pad(x)
  x = x[:, :, None]
  return x

def data_split(ecgs, labels, train_frac):
  dataset = []
  for ecg, label in zip(ecgs, labels):
    dataset.append((ecg, label))

  train_cut = int(train_frac*len(dataset))
  random.shuffle(dataset)
  train = dataset[:train_cut]
  dev   = dataset[train_cut:]

  train_ecg   = []
  train_label = []
  for ecg, label in train:
    train_ecg.append(ecg)
    train_label.append(label)

  dev_ecg   = []
  dev_label = []
  for ecg, label in dev:
    dev_ecg.append(ecg)
    dev_label.append(label)

  return (train_ecg, train_label), (dev_ecg, dev_label)

