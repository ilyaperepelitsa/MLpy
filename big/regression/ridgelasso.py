from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1.0)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 1.0)

from sklearn.linear_model import ElasticNet
lasso = ElasticNet(alpha = 1.0, l1_ratio = 0.5)
# l1 = 1 makes it a lasso


import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header = None, sep = "\s+")


def lin_regplot(X, y, model):
    plt.scatter(X, y, c = "blue")
    plt.plot(X, model.predict(X), color = "red")
    return None

df.head()
df.columns = ["CRIM", "ZN", "INDUS", "CHAS",
              "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


X = df[["RM"]].values
y = df[["MEDV"]].values

from sklearn.tree import DecisionTreeRegressor
X = df[["LSTAT"]].values
y = df["MEDV"].values

tree = DecisionTreeRegressor(max_depth = 3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel("Price in $1000\'s [MEDV]")
plt.show()







X = df.iloc[: , :-1].values
y = df["MEDV"].values

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = 0.4, random_state = 1)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 1000, criterion = "mse", random_state = 1, n_jobs = -1)
forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print("MSE train: %.3f, test: %.3f" % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

print("R^2 train: %.3f, test: %.3f" % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


### looking at Residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "black",
            marker = "o", s = 35, alpha = 0.5, label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen",
            marker = "s", s = 35, alpha = 0.7, label = "Test data")

plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = "red")
plt.xlim([-10, 50])
plt.show()
