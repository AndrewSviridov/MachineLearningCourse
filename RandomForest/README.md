## SizeOfRandomForest

� ���� ������� ��� ����� ���������� �� ���������� �������� ���������� ���� � ����������� �� ���������� �������� � ���.

1. ��������� ������ �� ����� abalone.csv. ��� �������, � ������� ��������� ����������� ������� ������� (����� �����) �� ���������� ����������.

2. ������������ ������� Sex � ��������: �������� F ������ ������� � -1, I � � 0, M � � 1. ���� �� ����������� Pandas, �� �������� ��������� ���: data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

3. ��������� ���������� ������ �� �������� � ������� ����������. � ��������� ������� �������� ������� ����������, � ��������� � ��������.

4. ������� ��������� ��� (sklearn.ensemble.RandomForestRegressor) � ��������� ������ ��������: �� 1 �� 50 (�� �������� ��������� "random_state=1" � ������������). ��� ������� �� ��������� ������� �������� ������ ����������� ���� �� �����-��������� �� 5 ������. ����������� ��������� "random_state=1" � "shuffle=True" ��� �������� ���������� �����-��������� sklearn.cross_validation.KFold. � �������� ���� �������� �������������� ������������� ������������ (sklearn.metrics.r2_score).

5. ����������, ��� ����� ����������� ���������� �������� ��������� ��� ���������� �������� �� �����-��������� ���� 0.52. ��� ���������� � ����� ������� �� �������.

6. �������� �������� �� ��������� �������� �� ���� ����� ����� ��������. ���������� �� ���?

����� �� ������ ������� � ��������� ����, ���������� ����� � ������ �������. �������� ��������, ��� ������������ ����� �� ������ ��������� ������� ������ � �����. ������ ����� �������� ������������ ��������� Coursera. �� �������� ��� ���, ����� ������ ��� �����������.