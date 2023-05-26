#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Ayomide Ayowole-Obi
# Data Mining & Visualization 
# Assignment 7
# 4/18/23
# =============================================================================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# Encoding the Dependent Variable - species
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scale = sc.fit_transform(X)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_scale, y)

# Predicting the Test set results
y_pred = classifier.predict(X_scale)

# Showing the Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y, y_pred))
print(accuracy_score(y, y_pred))

# =============================================================================
# RBF
# [[49  1  0]
#  [ 0 36 14]
#  [ 0 12 38]]
# 0.82
# =============================================================================

# Visualizing the dataset results
from matplotlib.colors import ListedColormap
X_set, y_set = X_scale, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.
             predict(np.array([X1.flatten(), X2.flatten()]).T).
             reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'gold')))
for i in np.unique(y_set):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                color = ListedColormap(('brown', 'palegreen', 'yellow'))(i),
                edgecolors = 'black')
plt.title('SVM (Training set)')
plt.xlabel('Sepal Width (Scaled)')
plt.ylabel('Sepal Length (Scaled)')
plt.show()




