## Metrics

1. ��������� ���� classification.csv. � ��� �������� �������� ������ �������� ������� (������� true) � ������ ���������� �������������� (������� pred).

2. ��������� ������� ������ �������������:

|                    | Actual Positive | Actual Negative |
|--------------------|-----------------|-----------------|
| Predicted Positive | TP              | FP              |
| Predicted Negative | FN              | TN              |

��� ����� ����������� �������� TP, FP, FN � TN �������� �� ������������. ��������, FP � ��� ���������� ��������, ������� ����� 0, �� ���������� ���������� � ������ 1. ����� � ������ ������� � ������ ����� ����� ������.

3. ���������� �������� ������� �������� ��������������:
* Accuracy (���� ����� ���������) � sklearn.metrics.accuracy_score
* Precision (��������) � sklearn.metrics.precision_score
* Recall (�������) � sklearn.metrics.recall_score
* F-���� � sklearn.metrics.f1_score

� �������� ������ ������� ��� ������ ����� ����� ������.

4. ������� ������ ��������� ��������������. � ����� scores.csv �������� �������� ������ � �������� ������� �������������� �������������� ������ ��� ������� �������������� �� ��������� �������:
* ��� ������������� ��������� � ����������� �������������� ������ (������� score_logreg),
* ��� SVM � ������ �� ����������� ����������� (������� score_svm),
* ��� ������������ ��������� � ���������� ����� ������� ������� (������� score_knn),
* ��� ��������� ������ � ���� ������������� �������� � ����� (������� score_tree).

��������� ���� ����.

5. ���������� ������� ��� ROC-������ ��� ������� ��������������. ����� ������������� ����� ���������� �������� ������� AUC-ROC (������� �������� �������)? �������������� �������� sklearn.metrics.roc_auc_score.

6. ����� ������������� ��������� ���������� �������� (Precision) ��� ������� (Recall) �� ����� 70% ?

����� �������� ����� �� ���� ������, ������� ��� ����� precision-recall-������ � ������� ������� sklearn.metrics.precision_recall_curve. ��� ���������� ��� �������: precision, recall, thresholds. � ��� �������� �������� � ������� ��� ������������ �������, ��������� � ������� thresholds. ������� ������������ �������� �������� ����� ��� �������, ��� ������� ������� �� ������, ��� 0.7.

���� ������� �������� ������� �����, �� ����� � ������� ����� ���������� �������������� ������, ��������, 0.42. ��� ������������� ���������� ������� ����� �� ���� ������.

����� �� ������ ������� � ��������� ����, ���������� ����� � ������ �������. �������� ��������, ��� ������������ ����� �� ������ ��������� ������� ������ � �����. ������ ����� �������� ������������ ��������� Coursera. �� �������� ��� ���, ����� ������ ��� �����������.