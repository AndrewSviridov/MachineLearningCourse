import pandas
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

# returns k with max accuracy
def accuracyKNN(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracy = []
    for i in range(1, 51):
        clf = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(estimator=clf, X=X, y=y, cv=kf)
        accuracy.append(np.mean(score))
    return accuracy.index(max(accuracy)) + 1, max(accuracy)


data = pandas.read_csv('wine.csv', names=['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                                          'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                                          'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'], index_col=None)

X = data.loc[:, ('Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline')]
y = data.loc[:, 'Class']

k, maxAccur = accuracyKNN(X, y)
print("k = {}, max accuracy = {} without feature normalization".format(k, maxAccur))

X_scaled = scale(X)
accuracy = []

k, maxAccur = accuracyKNN(X_scaled, y)
print("k = {}, max accuracy = {} with feature normalization".format(k, maxAccur))


