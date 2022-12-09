# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:53:25 2022

@author: Lunafernandavid
"""

from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion

if LooseVersion(sklearn_version) < LooseVersion('0.18'):
    raise ValueError('Please use scikit-learn 0.18 or newer')
!pip install -U scikit-learn
#!pip install -U scikit-learn
from IPython.display import Image
%matplotlib inline
import numpy as np
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


tmp_1 = load_wine()

tmp_2 = pd.DataFrame(tmp_1['data'],
                     columns=tmp_1['feature_names'])

tmp_2['target_name'] = pd.Series(tmp_1['target'], name='target_values')

tmp_2.groupby(['target_name']).count()

tmp_2 = tmp_2.sort_values(by = ['target_name'])

tmp_2 = tmp_2.reset_index(drop = True)

tmp_2['target_name'].value_counts()

tmp_2.columns

X = tmp_2[['alcohol', 'total_phenols']]
y = tmp_2.target_name

print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))

print('Labels counts in y_train:', np.bincount(y_train))

print('Labels counts in y_test:', np.bincount(y_test))

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((y_train, y_test))

def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])


x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()

ax = plt.subplot(111)

for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
#plt.savefig('images/03_19.png', dpi=300, bbox_inches='tight')
plt.show()

#########
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=6, 
                              random_state=1)
tree.fit(X_train_std, y_train)

X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()

#####
tree.score(X_train_std, y_train)

y_pred = tree.predict(X_test_std)

print('Datos mal clasificados: %d' % (y_test != y_pred).sum())

print('Precisión: %.2f' % tree.score(X_test_std, y_test))

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=[0, 
                                        1,
                                        2],
                           feature_names=['alcohol', 
                                          'total_phenols'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('tree.png') 

tree2 = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=4, 
                              random_state=1)
tree2.fit(X_train_std, y_train)

X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree2, test_idx=range(105, 150))

plt.xlabel('alcohol')
plt.ylabel('total_phenols')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()

tree2.score(X_train_std, y_train)
y_pred = tree2.predict(X_test_std)
print('Datos mal clasificados: %d' % (y_test != y_pred).sum())
print('Precisión: %.2f' % tree2.score(X_test_std, y_test))

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data2 = export_graphviz(tree2,
                           filled=True, 
                           rounded=True,
                           class_names=[0, 
                                        1,
                                        2],
                           feature_names=['alcohol', 
                                          'total_phenols'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data2) 
graph.write_png('tree2.png') 


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=100, 
                                random_state=1,
                                n_jobs=4)
forest.fit(X_train_std, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('alcohol')
plt.ylabel('total_phenols')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

forest.score(X_train_std, y_train)

y_pred = forest.predict(X_test_std)
print('Datos mal clasificados: %d' % (y_test != y_pred).sum())
print('Precisión: %.2f' % forest.score(X_test_std, y_test))

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data3 = export_graphviz(forest,
                           filled=True, 
                           rounded=True,
                           class_names=[0, 
                                        1,
                                        2],
                           feature_names=['alcohol', 
                                          'total_phenols'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data3) 
graph.write_png('tree3.png') 