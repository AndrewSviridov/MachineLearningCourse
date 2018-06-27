import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data = pandas.read_csv('titanic.csv', index_col=None)
data = data.loc[:, ('Pclass', 'Fare', 'Age', 'Sex', 'Survived')]
data = data.dropna(how='any')

y = data.loc[:, 'Survived']
X = data.loc[:, ('Pclass', 'Fare', 'Age', 'Sex')]

sex_map = {'male': 0, 'female': 1}
X.Sex = X.Sex.map(sex_map)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = (-clf.feature_importances_).argsort()[:2]
print("Most important features: {} and {}".format(X.columns[importances[0]], X.columns[importances[1]]))