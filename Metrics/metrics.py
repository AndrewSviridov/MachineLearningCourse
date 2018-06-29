from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import pandas
import numpy as np


def max_precision(precision, recall):
    over70 = []

    for rec in np.nonzero(recall >= 0.7):
        over70.append(precision[rec])

    return np.max(over70)


classification = pandas.read_csv("classification.csv", index_col=None)
scores = pandas.read_csv("scores.csv", index_col=None)

error_matrix = {
    'TP': 0,
    'TN': 0,
    'FP': 0,
    'FN': 0
}

for i in range(classification.__len__()):
    if classification.values[:, 0][i] == 1:
        if classification.values[:, 1][i] == 1:
            error_matrix['TP'] += 1
        else:
            error_matrix['FN'] += 1
    else:
        if classification.values[:, 1][i] == 1:
            error_matrix['FP'] += 1
        else:
            error_matrix['TN'] += 1

print("TP = {}".format(error_matrix['TP']))
print("TN = {}".format(error_matrix['TN']))
print("FP = {}".format(error_matrix['FP']))
print("FN = {}".format(error_matrix['FN']))
print("accuracy = %.2f" % accuracy_score(classification.values[:, 0], classification.values[:, 1]))
print("precision = %.2f" % precision_score(classification.values[:, 0], classification.values[:, 1]))
print("recall = %.2f" % recall_score(classification.values[:, 0], classification.values[:, 1]))
print("f1_score = %.2f" % f1_score(classification.values[:, 0], classification.values[:, 1]))

print("roc_auc_score_logreg = %.2f" % roc_auc_score(scores.values[:, 0], scores.values[:, 1]))
print("roc_auc_score_svm = %.2f" % roc_auc_score(scores.values[:, 0], scores.values[:, 2]))
print("roc_auc_score_knn = %.2f" % roc_auc_score(scores.values[:, 0], scores.values[:, 3]))
print("roc_auc_score_tree = %.2f" % roc_auc_score(scores.values[:, 0], scores.values[:, 4]))

pr, r, _ = precision_recall_curve(scores.values[:, 0], scores.values[:, 1])
print("max_precision of logreg = %.2f (recall not less than 0.7)" % max_precision(pr, r))

pr, r, _ = precision_recall_curve(scores.values[:, 0], scores.values[:, 2])
print("max_precision of svm = %.2f (recall not less than 0.7)" % max_precision(pr, r))

pr, r, _ = precision_recall_curve(scores.values[:, 0], scores.values[:, 3])
print("max_precision of knn = %.2f (recall not less than 0.7)" % max_precision(pr, r))

pr, r, _ = precision_recall_curve(scores.values[:, 0], scores.values[:, 4])
print("max_precision of tree = %.2f (recall not less than 0.7)" % max_precision(pr, r))


