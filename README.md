# Sklearn-Crossvalidation

import sklearn
from sklearn import datasets
 
## load iris dataset
iris = datasets.load_iris()

## divided atribut and target, assign into variable x, y
x=iris.data
y=iris.target

from sklearn.model_selection import cross_val_score
from sklearn import tree
 
clf = tree.DecisionTreeClassifier()

## evaluation model perform with cross val score
scores = cross_val_score(clf, x, y, cv=5)
scores

array([0.96666667, 0.96666667, 0.9       , 0.93333333, 1.        ])
