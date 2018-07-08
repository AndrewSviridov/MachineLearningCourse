from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas
import numpy as np
import matplotlib.pyplot as plt

data = pandas.read_csv("gbm-data.csv")
y = data.values[:, 0]
X = data.values[:, 1:data.columns.__len__()]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


def sigmoid(y):
    return 1. / (1 + np.exp(-y))


for i in [1, 0.5, 0.3, 0.2, 0.1]:
    gbt = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=i)
    gbt.fit(X_train, y_train)

    train_loss = []
    test_loss = []

    for j, y_pred in enumerate(gbt.staged_decision_function(X_train)):
        train_loss.append(log_loss(y_train, sigmoid(y_pred)))

    for j, y_pred in enumerate(gbt.staged_decision_function(X_test)):
        test_loss.append(log_loss(y_test, sigmoid(y_pred)))

    min_train_loss = np.min(train_loss)
    iter_train = np.argmin(train_loss)
    min_test_loss = np.min(test_loss)
    iter_test = np.argmin(test_loss)

    print("{}:\nmin train_loss {} on iteration {}".format(gbt, min_train_loss, iter_train))
    print("{}:\nmin test_loss {} on iteration {}".format(gbt, min_test_loss, iter_test))

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

rfc = RandomForestClassifier(n_estimators=iter_test, random_state=241)
rfc.fit(X_train, y_train)
y_pred = rfc.predict_proba(X_test)
print("Random Forest Classifier log_loss {}".format(log_loss(y_test, y_pred)))