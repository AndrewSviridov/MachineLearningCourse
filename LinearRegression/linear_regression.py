import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
import numpy as np

data_train = pandas.read_csv("salary-train.csv")
data_test = pandas.read_csv("salary-test-mini.csv")

# texts to lowercase and replacing everything, except letters and numbers, with spaces
def preprocessing(data):
    for key in data.keys():
        encoded_key = key.encode('utf-8')
        if encoded_key != 'SalaryNormalized':
            data[encoded_key] = data[encoded_key].str.lower()
        data[encoded_key] = data[encoded_key].replace('[^a-zA-Z0-9]', ' ', regex=True)

    return data

data_train = preprocessing(data_train)
data_test = preprocessing(data_test)

vectorizer = TfidfVectorizer(min_df=5)
fd_train = vectorizer.fit_transform(data_train['FullDescription'])
fd_test = vectorizer.transform(data_test['FullDescription'])

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_categ_train = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_categ_test = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

features_train = hstack((fd_train, X_categ_train))
X_train = hstack((features_train, np.array(data_train['SalaryNormalized'])[:,None]))

X_test = hstack((fd_test, X_categ_test))

print(X_train)
print(X_test)

clf = Ridge(alpha=1, random_state=241)
clf.fit(features_train, data_train['SalaryNormalized'])

predicted = clf.predict(X_test)

print(predicted)