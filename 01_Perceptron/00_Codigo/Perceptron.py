# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:29:06 2022

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

tmp_2.groupby(['target_name']).count()

tmp_2 = tmp_2.sort_values(by = ['target_name'])

tmp_2 = tmp_2.reset_index(drop = True)

tmp_2.groupby(['target_name']).count()


#los niveles para la variable 'target_name' son los siguientes:
#0=="Maligno",1=="Benigno"



class Perceptron(object):
    def __init__(self, eta=0.01,n_iter=100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale=0.01, size = 1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update !=0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,-1)
                

#definir la variable a predecir
#en este caso solo tenemos dos niveles 0, 1 

y = tmp_2['target_name'].values

#vamos a entrenar el modelo con todas las variables menos 
#la variable target 

    
#tmp_2.columns.values[5]
#tmp_2.columns.values[9]

tmp_2.columns

#solo quitamos la variable target

X = tmp_2.drop(['target_name'], axis=1).values

ppn = Perceptron(eta = 0.1, n_iter = 100)

ppn.fit(X, y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker = 'o')
plt.show()

np.transpose(ppn.errors_)

##ahora veamos que sucede si normalizamos las variables y repetimos el proceso 
# de entrenamiento del perceptron

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/x.std(), axis = 0)

X_norm = mean_norm(tmp_2.drop(['target_name'], axis=1))

X_norm = X_norm.values

ppn = Perceptron(eta = 0.1, n_iter = 100)

ppn.fit(X_norm, y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker = 'o')
plt.show()

np.transpose(ppn.errors_)

#se ve un cambio en la prediccion asociado a el orden de los errores 
#pero de todas formas hay problemas que persisten en el tiempo 

tmp_2.groupby(['target_name']).count()

#esta es la clases de los 0
tmp_2.loc[211,'target_name']

#esta es la clase de los 1
tmp_2.loc[212,'target_name']

##usemos la ordenacion inducida por las clases para separar los 
#dos conjuntos y hacer las graficas 
#tomemos dos variables para hacer el analisis grafico 

tmp_2.columns

GR_1 = tmp_2.loc[:,['mean radius','worst texture']].values

plt.scatter(GR_1[:211,0], GR_1[:211,1], color ='red',
            marker = 'o',label = 'Maligno')

plt.scatter(GR_1[211:GR_1.shape[0],0],GR_1[211:GR_1.shape[0],1],
            marker = 'x',label = 'Benigno')

plt.xlabel('mean radius')
plt.ylabel('worst texture')
plt.legend(loc = 'upper left')
plt.show()


#entrenemos el perceptron solo con esas dos variables 
#grafiquemos los resultados de la prediccion 
#de modo que hacemos el mismo proceso pero supeditado
#a dos variables asi:
    
GR_1 = tmp_2.loc[:,['mean radius','worst texture','target_name']]

GR_NORM = mean_norm(GR_1.drop(['target_name'], axis=1))

y = GR_1['target_name'].values

GR_NORM = GR_NORM.values

ppn = Perceptron(eta = 0.1, n_iter = 100)

ppn.fit(GR_NORM, y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker = 'o')
plt.show()

np.transpose(ppn.errors_)

#del mismo modo hacer la grafica de la frontera de desicion

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
    
plot_decision_regions(GR_NORM, y, classifier=ppn)
plt.xlabel('mean radius')
plt.ylabel('worst texture')
plt.legend(loc = 'upper left')
plt.show()

#para este problema el perceptron no es un buen metodo a usar



tmp_2.groupby(['target_name']).count()

tmp_2['target_name'].value_counts().describe()


