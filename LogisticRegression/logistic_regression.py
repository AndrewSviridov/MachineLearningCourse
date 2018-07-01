import pandas
from sklearn.linear_model import LogisticRegression
import math
from sklearn.metrics import roc_auc_score
import numpy as np

# C -- regularization parameter, X -- data, k -- step size, eps -- calculation accuracy
def logisticRegression(C, X, k, eps):
    y = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]

    w1_k = 0
    w2_k = 0

    j = 0

    # gradient descent (max iterations = 1e4)
    while j < 10000:
        w1_k1 = w1_k + k * np.mean((y * x1) / (1 + np.exp(y * (w1_k * x1 + w2_k * x2)))) - k * C * w1_k
        w2_k1 = w2_k + k * np.mean((y * x2) / (1 + np.exp(y * (w1_k * x1 + w2_k * x2)))) - k * C * w2_k

        # Euclidean distance between weight vectors on neighboring iterations must be no greater than 1e-5
        if math.sqrt((w1_k - w1_k1) ** 2 + (w2_k - w2_k1) ** 2) < eps:
            break
        else:
            j += 1
        w1_k, w2_k = w1_k1, w2_k1

    sigmoid = 1. / (1 + np.exp(-w1_k * x1 - w2_k * x2))

    print("AUC-ROC = %.3f" % roc_auc_score(y, sigmoid))
    print("Number of iterations = {}".format(j))


data = pandas.read_csv('data-logistic.csv', index_col=None)

X = data.values

# logistic regression without regularization
print("Logistic regression without regularization")
logisticRegression(0, X, 0.1, 0.00001)

# logistic regression with L2-regularization
print("Logistic regression with L2-regularization")
logisticRegression(10, X, 0.1, 0.00001)
