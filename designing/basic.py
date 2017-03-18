import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def normal(mean = 0, var = 1):
    sigma = np.sqrt(var)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, mlab.normpdf(x, mean, sigma))
    plt.show()

normal(1, 0.5)




from scipy.stats import binom
def binomial(x = 10, n = 10, p = 0.5):
    fig, ax = plt.subplots(1, 1)
    x = range(x)
    rv = binom(n, p)
    plt.vlines(x, 0, (rv.pmf(x)), colors = "k", linestyles = "-")
    plt.show()

binomial()



# from scipy.stats import poisson
# def pois(x = 1000):
#     xr = range(x)
#     ps = poisson(xr)
#     plt.plot(ps.pmf(x/2))
#
# pois()

import scipy.stats as stats
def cdf(s1 = 50, s2 = 0.2):
    x = np.linspace(0, s2 * 100, s1 * 2)
    cd = stats.binom.cdf
    plt.plot(x, cd(x, s1, s2))
    plt.show()

cdf()





####       GRADIENT DESCENT THINGY
import numpy as np
import random
import matplotlib.pyplot as plt

def gradientDescent(x, y, alpha, numIterations):
    xTrans = x.transpose()
    # print("Xtras = " + xTrans)
    m, n = np.shape(x)
    theta = np.ones(n)
    for i in range(0, numIterations):
        hwx = np.dot(x, theta)
        loss = hwx - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f " % (i, cost))
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape = (numPoints, 2))
    # print(x)
    y = np.zeros(shape = numPoints)
    # print(y)
    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


def plotData(x, y, theta):
    plt.scatter(x[..., 1], y)
    plt.plot(x[..., 1], [theta[0] + theta[1] * xi for xi in x[..., 1]])
    plt.show()


x, y = genData(20, 25, 10)
iterations = 10000
alpha = 0.001
theta = gradientDescent(x, y, alpha, iterations)
plotData(x, y, theta)




               
