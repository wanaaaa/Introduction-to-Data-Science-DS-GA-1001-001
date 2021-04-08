import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# from a0Functions import *

trainData = np.loadtxt('trainNum.txt')
# testData = np.loadtxt('testNum.txt')

np.random.shuffle(trainData)

X_train = trainData[:, :-1]
y_train = trainData[:, -1]; y_train = y_train.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
grid={"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty":["l1","l2"]}
logReg = linear_model.LogisticRegression()
logReg_cv = GridSearchCV(logReg, grid, cv=10)
logReg_cv.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",logReg_cv.best_params_)
print("accuracy :",logReg_cv.best_score_)

# grid = {'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
# treeClf = DecisionTreeClassifier()
# treeClf_cv = GridSearchCV(treeClf, grid)
# treeClf_cv.fit(X_train, y_train)
# print("tuned hpyerparameters :(best parameters) ", treeClf_cv.best_params_)
# print("accuracy :", treeClf_cv.best_score_)

# grid = {'C':[1,10,100,1000],'gamma':[0.0001, 0.001, 0.01, 0.1, 0, 1], 'kernel':['linear','rbf']}
# svcClf = SVC()
# svcClf_cv = GridSearchCV(svcClf, grid, refit= True, verbose= 2)
# svcClf_cv.fit(X_train, y_train)
# print("tuned hpyerparameters :(best parameters) ",svcClf_cv.best_params_)
# print("accuracy :",svcClf_cv.best_score_)

