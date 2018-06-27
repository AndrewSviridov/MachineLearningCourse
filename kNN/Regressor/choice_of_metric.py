import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

boston = load_boston()
X, y = boston.data, boston.target
scaled_X = scale(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy = list()
for i in np.linspace(start=1, stop=10, num=200):
    knr = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=i)
    score = cross_val_score(estimator=knr, X=scaled_X, y=y, scoring='neg_mean_squared_error', cv=kf)
    accuracy.append(np.mean(score))

print("p = {}, max mean_squared_error = {}".format(accuracy.index(max(accuracy))+1, max(accuracy)))

