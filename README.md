# Sklearn-Crossvalidation

using cross_validation_score in the decision_tree classifier. The dataset used is the iris dataset.

- import sklearn
- from sklearn import datasets
 
## Load Iris Dataset
The dataset we will use is the iris dataset. Then we divide the attributes and labels in the dataset.
- iris = datasets.load_iris()

## Divided Atribut and Target, Assign into Variable x, y
We will create our first machine learning model, such as a decision tree, using the Scikit Learn library. Machine learning models are also often referred to as classifiers. Furthermore, the variable clf stands for classifier.

- x=iris.data
- y=iris.target

- from sklearn.model_selection import cross_val_score
- from sklearn import tree
 
- clf = tree.DecisionTreeClassifier()

## Evaluation Model Perform with Cross Validation Score
Once the dataset and model are ready, we can use cross validation to evaluate the performance of the machine learning model. The cross_val_score() function as below accepts 4 parameters, namely, 'clf' which is the machine learning model, 'X' which is the attribute of the dataset, 'y' which is the label of the dataset, and 'cv' which is the number of folds that will be used in cross validation.
- scores = cross_val_score(clf, x, y, cv=5)
- scores

- array([0.96666667, 0.96666667, 0.9       , 0.93333333, 1.        ])

Cross_val_score returns a value in the form of an array consisting of the test accuracy for each fold of the dataset. To print and find out the results, add the scores code below the previous code. It looks like the image below.

![image](https://github.com/diantyapitaloka/Sklearn-Crossvalidation/assets/147487436/a9517cd8-0fc0-4bdd-93ff-8634c1ccae81)

