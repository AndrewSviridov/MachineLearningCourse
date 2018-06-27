from __future__ import print_function
import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

#
print("Number of people on the ship:")
print("Male: %d" % data[data['Sex'] == 'male'].Sex.count())
print("Female: %d" % data[data['Sex'] == 'female'].Sex.count())

print("Survived passengers in percentage: %.2f" % (
    (float(data[data['Survived'] == 1].Survived.count()) / float(data.Survived.count())) * 100))

print("1 class passengers in percentage: %.2f" % (
    float(data[data['Pclass'] == 1].Pclass.count()) / float(data.Pclass.count()) * 100))

print("Average age of passengers: %.2f" % (data['Age'].mean()))
print("Median age of passengers: %.2f" % (data['Age'].median()))

print("Pearson's correlation between SibSp and Parch %.4f" % (data.SibSp.corr(data.Parch, method='pearson')))

data['FirstName'] = data['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]

popularName = data.loc[(data['Sex'] == 'female'), 'FirstName'].mode()

print("Most popular female name: {}".format(popularName))
