from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    --------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
    Tuning parameter of the RBF kernel

    n_components: int
    Number of principal components to return

    Returns
    ---------------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset

    """

    # Calculate pariwise squared Euclidian distances
    # in the MxN dimensional datasets
    sq_dists = pdist(X, "sqeuclidean")

    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    #Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = N - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    #Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    return(X_pc)

##### HALF-MOON SHAPES

from sklearn.datasets import make_moons
X, y = make_moons(n_samples = 100, random_state = 123)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "red", marker = "^", alpha = 0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "blue", marker = "o", alpha = 0.5)
plt.show()


from sklearn.decomposition import PCA
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color = "red", marker = "^", alpha = 0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color = "blue", marker = "o", alpha = 0.5)
ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02, color = "red", marker = "^", alpha = 0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02, color = "blue", marker = "o", alpha = 0.5)

ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()





from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma = 15, n_components = 2)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color = "red", marker = "^", alpha = 0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color = "blue", marker = "o", alpha = 0.5)
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02, color = "red", marker = "^", alpha = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02, color = "blue", marker = "o", alpha = 0.5)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()










#### CIRCLES

from sklearn.datasets import make_circles
X, y = make_circles(n_samples = 1000, random_state = 123, noise = 0.2, factor = 0.2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "red", marker = "^", alpha = 0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "blue", marker = "o", alpha = 0.5)
plt.show()


scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color = "red", marker = "^", alpha = 0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color = "blue", marker = "o", alpha = 0.5)
ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02, color = 'red', marker = "^", alpha = 0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02, color = "blue", marker = "o", alpha = 0.5)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()



X_kpca = rbf_kernel_pca(X, gamma = 15, n_components = 2)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color = "red", marker = "^", alpha = 0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color = "blue", marker = "o", alpha = 0.5)
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02, color = 'red', marker = "^", alpha = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color = "blue", marker = "o", alpha = 0.5)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
# ax[1].set_xlim([-0.05, -0.01])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()





from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    --------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
    Tuning parameter of the RBF kernel

    n_components: int
    Number of principal components to return

    Returns
    ---------------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset

    lambdas: list
        Eigenvalues
    """

    # Calculate pariwise squared Euclidian distances
    # in the MxN dimensional datasets
    sq_dists = pdist(X, "sqeuclidean")

    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    #Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = N - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    #Obtaining eigenpairs from the centered kernel matrix
    #numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    #Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, -i] for i in range (1, n_components + 1)))

    #Collect the corresponding Eigenvalues
    lambdas = [eigvals[-i] for i in range (1, n_components + 1)]

    return alphas, lambdas

X, y = make_moons(n_samples = 100, random_state = 123)
alphas, lambdas = rbf_kernel_pca(X, gamma = 15, n_components = 1)

x_new = X[25]
x_new
x_proj = alphas[25]
x_proj

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/lambdas)

x_reproj = project_x(x_new, X, gamma = 15, alphas = alphas, lambdas = lambdas)
x_reproj


plt.scatter(alphas[y == 0, 0], np.zeros((50)), color = "red", marker = "^", alpha = 0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)), color = "blue", marker = "o", alpha = 0.5)
plt.scatter(x_proj, 0, color = 'black', label = "Original projection of pointX[25]", marker = "^", s = 100)
plt.scatter(x_proj, 0, color = "green", label = "Remapped pointX[25]", marker = "x", s = 500)
plt.legend(scatterpoints = 1)
plt.show()






##### SCIKIT SOLUTION

from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples = 100, random_state = 123, noise = 0.2)
scikit_kpca = KernelPCA(n_components = 2, kernel = "rbf", gamma = 1)
X_skernpca = scikit_kpca.fit_transform(X)


plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], color = "red", marker = "^", alpha = 0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], color = "blue", marker = "o", alpha = 0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
