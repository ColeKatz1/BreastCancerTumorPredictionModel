import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
tumorData = pd.read_csv("data.csv")
# clean up data

tumorData = tumorData.dropna(axis = 1)

# distribution of benign vs malignant 

sns.countplot(x = "diagnosis", data = tumorData)
plt.show()

# assign variables to x and y
y = tumorData["diagnosis"]
x = tumorData.drop(["diagnosis", "id"], axis = 1)

# correlation between variables

plt.figure(figsize=(20,15))
sns.heatmap(x.corr(), annot = True, cmap="coolwarm")
plt.show() 

# make model simpler by removing highly correlated variables (remove one of the two highly correlated variables) 

x = x.drop(["area_mean","radius_worst","perimeter_worst","area_worst"], axis = 1)

# graph of each variable

for i in x:
    sns.displot(tumorData, x=i, hue = "diagnosis", kind="kde", multiple="stack")
    plt.show()

# transform M/B to 0/1 - switch from categorical to numerical
le = LabelEncoder()
y= le.fit_transform(y)

# train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 10)

# logistic regression
log = LogisticRegression(random_state = 0)
log_model = log.fit(x_train, y_train)
y_pred = log.predict(x_test)
print('Logistic Regression:', log_model.score(x_test, y_test))
print('Logistic Predictions:', y_pred)
print('Actual Results', y_test)

# SVC linear

svc_lin = SVC(kernel = 'linear', random_state = 0)
svc_lin_model = svc_lin.fit(x_train, y_train)
print('SVC Linear:', svc_lin_model.score(x_test, y_test))

# SVC rbf

svc_rbf = SVC(kernel = 'rbf', random_state = 0)
svc_rbf_model = svc_rbf.fit(x_train, y_train)
print('SVC rbf:', svc_rbf_model.score(x_test, y_test))

# decision tree

tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree_model = tree.fit(x_train, y_train)
print('Decision Tree:', tree_model.score(x_test, y_test))

# random forest

forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest_model = forest.fit(x_train, y_train)
print('Random Forest:', forest_model.score(x_test, y_test))
