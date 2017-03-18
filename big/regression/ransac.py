from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(), max_trials = 100, min_samples = 50,
                        residual_metric = lambda x: np.sum(np.abs(x), axis = 1),
                        residual_threshold = 5.0, random_state = 0)



import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header = None, sep = "\s+")

df.head()
df.columns = ["CRIM", "ZN", "INDUS", "CHAS",
              "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


X = df[["RM"]].values
y = df[["MEDV"]].values
ransac.fit(X, y)


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c = "blue", marker = "o", label = "Inliers")
plt.scatter(X[outlier_mask], y[outlier_mask], c = "lightgreen", marker = "s", label = "Outliers")
plt.plot(line_X, line_y_ransac, color = "red")
plt.xlabel("Average number of rooms [RM]")
plt.ylabel("Price in $1000\'s [MEDV]")
plt.legend(loc = "upper left")
plt.show()


from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df ["MEDV"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "o", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Test data")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = "red")
plt.xlim([-10, 50])
plt.show()



from sklearn.metrics import mean_squared_error
print("MSE train: %.3f, test: %.3f" % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))

from sklearn.metrics import r2_score
print("R^2 train: %.3f, test: %.3f" %
        (r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
