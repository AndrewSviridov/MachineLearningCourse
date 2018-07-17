import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import datetime
np.set_printoptions(threshold=100)


def gradient_boosting(n_estimators, X, y, max_depth=3):
    start_time = datetime.datetime.now()

    gbc = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=241)
    kf = KFold(n_splits=5, shuffle=True)

    roc_score = np.empty([kf.n_splits, 1])

    # cross-validation
    for k, (train, test) in enumerate(kf.split(X, y)):
        x_train, y_train = X.iloc[train], y.iloc[train]
        x, y_test = X.iloc[test], y.iloc[test]

        gbc.fit(x_train, y_train)
        pred = gbc.predict_proba(x)[:, 1]

        # saving roc_score for each folds
        roc_score[k] = roc_auc_score(y_test, pred)

    print("n_estimators = {}\nAUC-ROC = {}".format(gbc.n_estimators, np.mean(roc_score))) # roc_score as mean of each fold
    print('Time elapsed:', (datetime.datetime.now() - start_time).seconds)

    # fit classifier on full train set
    gbc.fit(X, y)

    return gbc


def search_c(X, y):
    param_grid = {'C': [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]}
    scoring = {'AUC': "roc_auc"}

    kf = KFold(n_splits=5, shuffle=True)

    # normalization of features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # searching best regularization parameter by GridSearch
    clf = GridSearchCV(LogisticRegression(random_state=241), param_grid=param_grid, scoring=scoring, refit='AUC', cv=kf)
    clf.fit(X, y)

    print("Best params: {}".format(clf.best_params_))

    return clf.best_params_.values()[0]


def logistic_regression(X, y, C):
    start_time = datetime.datetime.now()

    scaler = StandardScaler()

    clf = LogisticRegression(C=C, random_state=241)
    kf = KFold(n_splits=5, shuffle=True)

    roc_score = np.empty([kf.n_splits, 1])

    # cross-validation
    for k, (train, test) in enumerate(kf.split(X, y)):

        # scaling each set
        x_train, y_train = scaler.fit_transform(X.iloc[train]), y.iloc[train]
        x_test, y_test = scaler.transform(X.iloc[test]), y.iloc[test]

        clf.fit(x_train, y_train)
        pred = clf.predict_proba(x_test)[:, 1]

        # saving roc_score for each folds
        roc_score[k] = roc_auc_score(y_test, pred)

    print("AUC-ROC = {}".format(np.mean(roc_score))) # roc_score as mean of each fold
    print('Time elapsed:', (datetime.datetime.now() - start_time).seconds)

    # scaling full train set
    X = scaler.fit_transform(X)

    # fit classifier on full train set
    clf.fit(X, y)

    return clf, scaler


def number_of_unique_heroes(X):
    N = pandas.unique(X[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                         'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']].values.ravel('K')).size
    print("Number of unique hero ID: {}".format(N))

    return N


def convert_categorical_features(N, X):
    # N -- number of different heros
    X_pick = np.zeros((X.shape[0], N))

    for i, match_id in enumerate(X.index):
        for p in xrange(5):
            X_pick[i, int(X.loc[match_id, 'r%d_hero' % (p + 1)] - 1)] = 1
            X_pick[i, int(X.loc[match_id, 'd%d_hero' % (p + 1)] - 1)] = -1

    # creating keys for dict
    keys = []
    for i in xrange(1, N + 1):
        keys.append('hero_%d' % i)

    # creating dict with None values
    heroes = dict.fromkeys(keys)

    # adding empty arrays as values into dict
    for hero in heroes.keys():
        heroes[hero] = []

    # filling arrays by X_pick values
    for j in range(X_pick.__len__()):
        for i in range(1, N + 1):
            heroes['hero_%d' % i].append(X_pick[j][i - 1])

    # creating dataframe from dict
    features_heroes = pandas.DataFrame(heroes, index=X.index)

    return features_heroes


features = pandas.read_csv('./features.csv', index_col='match_id')
X_test = pandas.read_csv('./features_test.csv', index_col='match_id')

# drop all features that can "look into the future" from train set
X_train = features.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                         'barracks_status_radiant', 'barracks_status_dire'], axis=1)
# target feature
y = features['radiant_win']

# checking for missing values in train set
for column in X_train:
    print("Column {}: number of missing values = {}".format(column, X_train.values.__len__() - X_train[column].count()))

# fill NA in train and test sets with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

gradient_boosting(10, X_train, y)
gradient_boosting(20, X_train, y)
gradient_boosting(30, X_train, y)
gradient_boosting(40, X_train, y)
gradient_boosting(50, X_train, y)
gradient_boosting(60, X_train, y)
gradient_boosting(70, X_train, y)
gradient_boosting(80, X_train, y)

# searching for best regularization parameter
best_C = search_c(X_train, y)
logistic_regression(X_train, y, best_C)

# exclude categorical features
cat_features = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']

new_X_train = X_train.drop(cat_features, axis=1)
new_X_test = X_test.drop(cat_features, axis=1)

# searching for best regularization parameter
best_C = search_c(new_X_train, y)
logistic_regression(new_X_train, y, best_C)

# renumbering cell values in range(N+1)
cells = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
         'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']

X_train[cells] = X_train[cells].rank(method='dense')
X_test[cells] = X_test[cells].rank(method='dense')


# checking for number of unique heroes in train and test sets
N_train = number_of_unique_heroes(X_train)
N_test = number_of_unique_heroes(X_test)

# convertation of categorical features in train and test sets
heroes_train = convert_categorical_features(N_train, X_train)
heroes_test = convert_categorical_features(N_test, X_test)

# concatenation converted features and rest set
new_X_train = pandas.concat([new_X_train, heroes_train], axis=1, join_axes=[new_X_train.index])
new_X_test = pandas.concat([new_X_test, heroes_test], axis=1, join_axes=[new_X_test.index])

# searching for best regularization parameter
best_C = search_c(new_X_train, y)
clf, scaler = logistic_regression(new_X_train, y, best_C)

# scaling test set
scaled_X = scaler.transform(new_X_test)

pred = clf.predict_proba(scaled_X)[:, 1]

print("min = {}".format(pred.min()))
print("max = {}".format(pred.max()))
