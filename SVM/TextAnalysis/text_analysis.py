from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(subset='all',categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# searching for best regularization parameter
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

best_C = gs.best_params_.values()[0]
print(best_C)

# fit SVM with the best C
clf = SVC(C=best_C, kernel='linear', random_state=241)
clf.fit(X, y)

feature_mapping = vectorizer.get_feature_names()

word_indicies = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
words = [feature_mapping[i].encode('utf-8') for i in word_indicies]

print("10 words with the highest absolute weight value:")
print(words.sort())

