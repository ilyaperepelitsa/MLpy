import numpy as np
a = np.arange(60).reshape(3, 4, 5)
a


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0., 5., 0.2)
x

plt.plot(x, x**4, "r", x, x*90, "bs", x, x**3, "g^")
plt.show()


# printing histograms

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(1000)
x

n, bins, patches = plt.hist(x, 10, normed = 1, facecolor = "g")
plt.xlabel("Frequency")
plt.ylabel("Probability")

plt.title("Histogram")
plt.text(40, 0.028, "mean = 100 std.dev. = 15")
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


N = 100
x = np.random.rand(N)
y = np.random.rand(N)

# colors = np.random.rand(N)
colors = ("r", "b", "g")
area = np.pi * (10 * np.random.rand(N)) ** 2
plt.scatter(x, y, s = area, c = colors, alpha = 0.5)
plt.show()


import pandas as pd
df = pd.read_csv("/Users/ilyaperepelitsa/Downloads/sampleData.csv")

df

df.head()


del df["Bureau of Meteorology station number"]
del df["Product code"]
del df["Days of accumulation of maximum temperature"]

df = df.rename(columns = {"Maximum temperature (Degree C) " : "maxtemp"})
df
df = df[(df.Quality == "Y")]
df



df.describe()

import matplotlib.pyplot as plt
plt.plot(df.Year, df.maxtemp)



from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

iris_X.shape
iris_y.shape


iris.DESCR





from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def knnDemo(X, y, n):

    #creates the classifier and fits it to the data
    res = 0.05
    k1 = knn(n_neighbors = n, p = 2, metric = "minkowski")
    k1.fit(X, y)

    # sets up the grid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res))


    # makes the prediction
    Z = k1.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # creates the color map
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    # plots the decision surface
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap_light)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plots the samples

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold)

    plt.show()




iris = datasets.load_iris()
X1 = iris.data[:, 0:3:2]
X2 = iris.data[:, 0:2]
X3 = iris.data[:, 1:3]


y = iris.target
knnDemo(X2, y, 15)




from sklearn.linear_model import Ridge
import numpy as np

def ridgeReg(alpha):
    n_samples, n_features = 10, 5
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)
    clf = Ridge(0.001)
    res = clf.fit(X, y)
    return res

res = ridgeReg(0.001)
print(res.coef_)




import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
np.random.seed()

X, y = make_circles(n_samples = 400, factor = 0.3, noise = 0.05)
kpca = KernelPCA(kernel = "rbf", gamma = 10)
X_kpca = kpca.fit_transform(X)
plt.figure()
plt.subplot(2, 2, 1, aspect = "equal")
plt.title("Original space")
reds = y == 0
blues = y == 1
plt.plot(X[reds, 0], X[reds, 1], "ro")
plt.plot(X[blues, 1], X[blues, 1], "bo")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(2, 2, 3, aspect = "equal")
plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro")
plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")

plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")
plt.subplots_adjust(0.02, 0.1, 0.98, 0.94, 0.04, 0.35)
plt.show()
# print("gamma = %0.2f" % gamma)





from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)
clf = svm.SVC(kernel = "linear", C = 1).fit(X_train, y_train)


scores = cross_validation.cross_val_score(clf, X_train, y_train, cv = 5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
