from sklearn.decomposition import PCA
import numpy as np
import pandas

close_prices = pandas.read_csv("close_prices.csv")
X = close_prices.drop(['date'], axis=1)

djia_index = pandas.read_csv("djia_index.csv")
djia_index = djia_index.drop(['date'], axis=1)

pca = PCA(.90)
pca.fit(X)

print("min components for 90% variance {}".format(pca.n_components_))

pca = PCA(n_components=10)
pca.fit(X)

# explained_variance_ratio = np.round(pca.explained_variance_ratio_, decimals=3)
# print("explained_variance_ratio_ {}".format(explained_variance_ratio))

pca.transform(X)
first_component = np.dot(X, pca.components_[0])

djia_index = np.array(djia_index).ravel()

print("Pearson correlation between first component and Dow Jones index = %.2f" % np.corrcoef(first_component, djia_index)[0, 1])

print("Company with the greatest weight {}".format(X.columns[np.argmax(pca.components_[0])]))

