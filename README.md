# Sklearn-Crossvalidation

using cross_validation_score in the decision_tree classifier. The dataset used is the iris dataset.

- import sklearn
- from sklearn import datasets
 
## Load Iris Dataset
iris = datasets.load_iris()

## Divided Atribut and Target, Assign into Variable x, y
- x=iris.data
- y=iris.target

- from sklearn.model_selection import cross_val_score
- from sklearn import tree
 
- clf = tree.DecisionTreeClassifier()

## Evaluation Model Perform with Cross Validation Score
- scores = cross_val_score(clf, x, y, cv=5)
- scores

- array([0.96666667, 0.96666667, 0.9       , 0.93333333, 1.        ])

Cross_val_score returns a value in the form of an array consisting of the test accuracy for each fold of the dataset. To print and find out the results, add the scores code below the previous code. It looks like the image below.

![image](https://github.com/diantyapitaloka/Sklearn-Crossvalidation/assets/147487436/a9517cd8-0fc0-4bdd-93ff-8634c1ccae81)

