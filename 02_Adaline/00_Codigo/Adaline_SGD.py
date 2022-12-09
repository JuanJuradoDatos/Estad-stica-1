# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:42:32 2022

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

#acoplamos la Adaline, con este metodo de actualizacion de los parametros 
#para tener una convergencia mas rapida y precisa del clasificador
tmp_1 = load_breast_cancer()

tmp_2 = pd.DataFrame(tmp_1['data'],
                     columns=tmp_1['feature_names'])

tmp_2['target_name'] = pd.Series(tmp_1['target'], name='target_values')

tmp_2.groupby(['target_name']).count()

#se selecciona la misma base pues en el codigo se esta trabajando solo 
#con una variable que clasifica en dos niveles posibles 

y = tmp_2['target_name'].values

X = tmp_2.drop(['target_name'], axis=1).values

class AdalineGD(object):
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta *X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2
            self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0,1,-1)
    
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        
ada1 = AdalineGD(n_iter=100, eta=0.001).fit(X,y)


ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.001')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')


plt.show()

#ahora haciendo el proceso de estandarizacion de los datos 

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/x.std(), axis = 0)

X_norm = mean_norm(tmp_2.drop(['target_name'], axis=1))

X_norm = X_norm.values

ada3 = AdalineGD(n_iter = 100, eta = 0.001)
ada3.fit(X_norm,y)        


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        
ax[0].plot(range(1, len(ada3.cost_) + 1), np.log10(ada3.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.001')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

#ahora con fines de graficar el plano de separacion 
#procedemos a dejar de nuevo dos variables y estandarizar 
#para tener una visual de la separacion 

#primero ordenamos con respecto a la etiqueta

tmp_2 = tmp_2.sort_values(by = ['target_name'])

GR_1 = tmp_2.loc[:,['mean radius','worst texture']].values

plt.scatter(GR_1[:211,0], GR_1[:211,1], color ='red',
            marker = 'o',label = 'Maligno')

plt.scatter(GR_1[211:GR_1.shape[0],0],GR_1[211:GR_1.shape[0],1],
            marker = 'x',label = 'Benigno')

plt.xlabel('mean radius')
plt.ylabel('worst texture')
plt.legend(loc = 'upper left')
plt.show()


GR_1 = tmp_2.loc[:,['mean radius','worst texture','target_name']]

GR_NORM = mean_norm(GR_1.drop(['target_name'], axis=1))

y = GR_1['target_name'].values

GR_NORM = GR_NORM.values

#de nuevo se construye la funcion de graficacion de la region de descicion

def plot_decision_regions(X,y,classifier,resolution=0.003):
    markers = ('A','B','C','D','E')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = GR_NORM[:,0].min() -1, GR_NORM[:,0].max() +1
    x2_min, x2_max = GR_NORM[:,1].min() -1, GR_NORM[:,1].max() +1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    print(xx1.shape)
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=GR_NORM[y == cl,0],
                    y=GR_NORM[y == cl,1],
                    alpha=0.8,
                    c = colors[idx],
                    label = cl,
                    edgecolor = 'black')

ada4 = AdalineGD(n_iter=100, eta=0.001)
ada4.fit(GR_NORM, y)

plot_decision_regions(GR_NORM, y, classifier=ada4)
plt.title('Adaline - Gradient Descent')
plt.xlabel('mean radius')
plt.ylabel('worst texture')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/02_14_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada4.cost_) + 1), ada4.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('images/02_14_2.png', dpi=300)
plt.show()

#aqui comienza el acople con el gradiente descendente

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch.

        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation""" # Funcion sigmoide y = 1/(1+exp(-w'*x)) , y = tangh(w'x)
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
##

ada = AdalineSGD(n_iter=10, eta=0.01, random_state=1)
ada.fit(GR_NORM, y)

plot_decision_regions(GR_NORM, y, classifier=ada)

plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('mean radius')
plt.ylabel('worst texture')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
plt.show()
