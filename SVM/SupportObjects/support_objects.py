from sklearn.svm import SVC
import pandas

data = pandas.read_csv('svm-data.csv', names=['Target', 'Feature1', 'Feature2'], sep=',')
X = data.loc[:, ('Target', 'Feature1', 'Feature2')]
y = data.loc[:, 'Target']

clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(X, y)

print("Support objects: {}".format(clf.support_+1))


