import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def accuracyPerceptron(X_train, y_train, X_test, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy


data_train = pandas.read_csv('perceptron_train.csv', names=['Target_variable', 'feature1', 'feature2'], index_col=None)
y_train = data_train.loc[:, 'Target_variable']
X_train = data_train.loc[:, ('feature1', 'feature2')]

data_test = pandas.read_csv('perceptron_test.csv', names=['Target_variable', 'feature1', 'feature2'], index_col=None)
y_test = data_test.loc[:, 'Target_variable']
X_test = data_test.loc[:, ('feature1', 'feature2')]

accuracy = accuracyPerceptron(X_train, y_train, X_test, y_test)
print("Quality of classifier without normalization: {}".format(accuracy))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

accur_scaled = accuracyPerceptron(X_train_scaled, y_train, X_test_scaled, y_test)
print("Quality of classifier after normalization: ", accur_scaled)

print("Difference between accuracies = {}".format(accur_scaled - accuracy))
