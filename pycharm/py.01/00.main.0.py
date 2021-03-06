import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split

# from a0Functions import *

trainData = np.loadtxt('trainNum.txt')
# testData = np.loadtxt('testNum.txt')

np.random.shuffle(trainData)

X_train = trainData[:, :-1]
y_train = trainData[:, -1]; y_train = y_train.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

# logReg = linear_model.LogisticRegression()
# logReg.fit(X_train, y_train)
# score = logReg.score(X_test, y_test)
# print(score)
# y_hat = logReg.predict(X_test)

# treeClf = DecisionTreeClassifier()
# treeClf.fit(X_train, y_train)
# y_hat = treeClf.predict(X_test)

# svcClf = SVC(gamma='auto')
# svcClf.fit(X_train, y_train)
# y_hat = svcClf.predict(X_test)

# adaBoostClf = AdaBoostClassifier()
# adaBoostClf.fit(X_train, y_train)
# y_hat = adaBoostClf.predict(X_test)
# score = adaBoostClf.score(X_test, y_test)
# print(score)

ranForestClf = RandomForestClassifier()
ranForestClf.fit(X_train, y_train)
score = ranForestClf.score(X_test, y_test)
print(score)
# y_hat = ranForestClf.predict(X_test)
# print(y_hat)

# totalRow = 0
# correctPrediction = 0;
# for i, yPredict in enumerate(y_hat):
#     if(y_test[i] == yPredict):
#         correctPrediction += 1
#     totalRow += 1
#
# print(correctPrediction/totalRow * 100)