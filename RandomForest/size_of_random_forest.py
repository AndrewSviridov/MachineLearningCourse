from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas
import numpy as np

data = pandas.read_csv("abalone.csv")
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = data.values[:, data.columns.__len__()-1]
X = data.values[:, 0: data.columns.__len__()-1]

kf = KFold(n_splits=5, shuffle=True, random_state=1)

for i in range(1, 50):
    regr = RandomForestRegressor(n_estimators=i, random_state=1)
    regr.fit(X, y)
    scores = cross_val_score(regr, X, y, cv=kf, scoring='r2')
    score = np.mean(scores)
    if score > 0.52:
        print("Min number of trees = {}:\nr2_score = {}".format(regr.n_estimators, score))
        break
