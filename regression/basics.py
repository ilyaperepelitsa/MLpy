import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header = None, sep = "\s+")

df.head()
df.columns = ["CRIM", "ZN", "INDUS", "CHAS",
              "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df.head()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style = "whitegrid", context = "notebook")
cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]
sns.pairplot(df[cols], size = 2.5)
plt.show()


import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True,
                 fmt = " .2f", annot_kws = {"size" : 15},
                 yticklabels = cols, xticklabels = cols)

plt.show()



# class LinearRegressionGD(object):
#
#     def __init__(self, eta = 0.001, n_iter = 20):
#         self.eta = eta
#         self.n_iter = n_iter
#
#     def fit(self, X, y):
#         self.w_ = np.zeros(1 + X.shape[1])
#         self.cost_ = []
#
#         for i in range(self.n_iter):
#             output = self.net_input(X)
#             errors = (y - output)
#             self.w_[1:] += (self.eta * X.T.dot(errors))
#             self.w_[0] += (self.eta * errors.sum())
#             cost = (errors ** 2).sum() / 2.0
#             self.cost_.append(cost)
#         return self
#
#     def net_input(self, X):
#         return np.dot(X, self.w_[1:]) + self.w_[0]
#
#     def predict(self, X):
#         return self.net_input(X)


X = df[["RM"]].values
y = df[["MEDV"]].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
X_std.shape
y_std = sc_y.fit_transform(y)
y_std.shape


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)


print("Slope: %.3f" % slr.coef_[0])
print("Intercept: %.3f" % slr.intercept_)
