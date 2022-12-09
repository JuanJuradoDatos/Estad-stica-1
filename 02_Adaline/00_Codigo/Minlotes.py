# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:12:32 2022

@author: Lunafernandavid
"""


import pandas as pd 
import os 
import sklearn as sk
from sklearn.datasets import load_breast_cancer
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

tmp_1 = load_breast_cancer()

tmp_2 = pd.DataFrame(tmp_1['data'],
                     columns=tmp_1['feature_names'])

tmp_2['target_name'] = pd.Series(tmp_1['target'], name='target_values')

y = tmp_2['target_name'].values

X = tmp_2.drop(['target_name'], axis=1).values

def create_batch_generator(X, y, batch_size = 128, shuffle = False):
  X_copy = np.array(X)
  y_copy = np.array(y)

  if shuffle:
    data = np.column_stack(X_copy, y_copy)
    #np.random.shuffle(data)
    #X_copy = data[:,:-1]
    #y_copy = data[:,-1].astype(int)
    random_state = 1
    rgen = np.random.RandomState(random_state)
    r = rgen.permutation(len(y_copy))
    X_copy = data[r, :-1]
    y_copy = data[r, -1].astype(int)

  for i in range(0, X.shape[0], batch_size):
    yield (X_copy[i:i+batch_size,:], y_copy[i:i+batch_size])

Lote = create_batch_generator(X, y, batch_size = 1, shuffle = True)

###segunda forma de hacer los lotes, por permutaciones

def batch_generator(X, y, batch_size = 128, shuffle = False, random_seed = None):

  X_copy = X.copy()
  y_copy = y.copy()

  idx = np.arange(y.shape[0])
  
  if shuffle:
    rng = np.random.RandomState(random_seed)
    rng.shuffle(idx)
    print(idx)

    X_copy = X[idx]
    y_copy = y[idx]

  for i in range(0, X.shape[0], batch_size):
    yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])

Lote =  batch_generator(X, y, batch_size = 32, shuffle = True, random_seed = 1)

for X_lote, y_lote in Lote:
  print(X_lote, y_lote)


