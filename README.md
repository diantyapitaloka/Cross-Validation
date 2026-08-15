## 🧊🍷🍵 Cross Validation 🍵🍷🧊
- We are using cross_validation_score in the decision_tree classifier. All the dataset used are the iris dataset below.
- Default Evaluation Metric: By default, the cross-validation score for a decision tree classification model calculates the overall accuracy of predictions on each test fold.
- Custom Performance Metrics: You can evaluate the model using alternative metrics—such as precision, recall, or F1-score—by specifying a custom scoring parameter.
- Automatic Class Balancing: For classification tasks, Scikit-Learn automatically applies Stratified K-Fold splitting, ensuring each fold contains a proportional representation of every target class.
- Multiple Output Scores: The cross-validation function returns a list of individual performance scores corresponding to each fold, rather than a single unified score.
- Statistical Summary: Averaging the scores gives the model's overall expected performance, while calculating the standard deviation reveals how stable or consistent the predictions are across different data samples.
- Data Shuffling and Randomization: Passing a customized splitter object allows you to shuffle the dataset before splitting, which prevents biased evaluations if the original data is sorted by label.
- Preservation of the Original Model: Cross-validation fits temporary internal copies of the classifier, meaning the original model instance remains unrained and unchanged after the evaluation process.
- Overfitting Detection: Comparing cross-validation results against the training data performance helps identify overfitting, which frequently occurs with fully grown, unrestricted decision trees.
- Multi-Metric and Timing Analysis: Upgrading to advanced cross-validation functions allows you to evaluate multiple metrics simultaneously while also tracking the time required for training and testing on each fold.
- Out-of-Sample Estimation: By testing exclusively on folds that were held out during each training iteration, cross-validation accurately simulates how well the Decision Tree will perform on unseen real-world data.
- Preprocessing Integration to Prevent Data Leakage: Feature engineering steps (such as missing value imputation or scaling) must be calculated exclusively on each training fold during cross-validation, preventing information from validation folds from corrupting the evaluation.
- Choice of $K$ and Bias-Variance Dynamics: Selecting the number of folds ($K$) creates a fundamental trade-off: lower fold counts (e.g., 2 folds) tend to underestimate performance due to smaller training subsets, whereas extreme fold counts (like leave-one-out) reduce performance bias but increase evaluation variance across iterations.
- Hyperparameter Optimization Alignment: Cross-validation serves as the evaluation engine for systematic parameter searches, allowing you to tune decision tree constraints (such as maximum tree depth or minimum leaf sizes) without overfitting to a static validation set.
- Group-Aware Splitting for Clustered Data: Standard cross-validation assumes each sample is independent; when datasets contain multiple repeated observations from the same subject or entity, group-aware splitting guarantees all records from a single entity remain strictly within either training or test folds.
- Sequential and Temporal Splitting: For time-dependent or sequential datasets, standard randomized K-fold violates chronological ordering; time-based cross-validation restricts model training strictly to historical data points to evaluate predictions on subsequent future folds.

```
import sklearn
from sklearn import datasets
```
 
## 🧊🍷🍵 Load Iris Dataset 🍵🍷🧊
The dataset we will use the iris dataset. Then we divide the attributes and labels on the many datasets.
```
iris = datasets.load_iris()
```

## 🧊🍷🍵 Divided Atribut and Target, Assign into Variable X, Y 🍵🍷🧊
We will create our first machine learning model, such as decision tree, also using the Scikit Learn library. Machine learning models are also often to referred to as classifiers. Furthermore, the variable clf stands for classifier.

```
x=iris.data
y=iris.target

from sklearn.model_selection import cross_val_score
from sklearn import tree
 
clf = tree.DecisionTreeClassifier()
```

## 🧊🍷🍵 Evaluation Model Perform with Cross Validation Score 🍵🍷🧊
Once the dataset and model are ready, we can use cross validation to evaluate the performance of the machine learning model. The cross_val_score() function as below accepts like 4 parameters, namely, 'clf' which is the machine learning model, 'X' which is the attribute of the dataset, 'y' which is the label of the dataset, and 'cv' which is the number of folds that will be used in cross validation.

```
scores = cross_val_score(clf, x, y, cv=5)
scores

array([0.96666667, 0.96666667, 0.9       , 0.93333333, 1.        ])
```

## 🧊🍷🍵 Output 🍵🍷🧊
Cross_val_score returns a value in the form of an array consisting of the test accuracy for each fold of the dataset. To print and find out the results, add the scores code below the previous code. It looks like the image below.

![image](https://github.com/diantyapitaloka/Sklearn-Crossvalidation/assets/147487436/a9517cd8-0fc0-4bdd-93ff-8634c1ccae81)

