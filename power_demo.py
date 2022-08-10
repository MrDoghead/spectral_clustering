import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import lobpcg
from algo import power_iter_for_second

def power_Fiedler(A, x, maxit=100):
    for _ in range(maxit):
        x = A @ x
        norm_factor = np.sum(x) / 128
        x = x - norm_factor 
    return x


rng = np.random.default_rng()

theta = np.linspace(0, 2 * np.pi, 64)
y = np.cos(theta)
z = np.sin(theta)

phi = np.pi/9
ring1 = np.array([y*np.sin(phi)+np.sin(phi) - .5,
                   y*np.cos(phi)+np.cos(phi) - .5, z]).T
phi = np.pi/8
ring2 = np.array([z, y*np.sin(phi)+np.sin(phi),
                   y*np.cos(phi)+np.cos(phi)]).T

data = np.concatenate((ring1, ring2))
data += 1e-1 * rng.standard_normal(data.shape)

data_scaled_to01 = (data - np.min(data))/(np.max(data) - np.min(data)) * 255
data_uint8 = data_scaled_to01.astype(np.uint8)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(data_uint8[:, 0], data_uint8[:, 1], data_uint8[:, 2], c=data_uint8 / 255, alpha=0.9, s=30)
plt.show()

img = rng.permutation(data_uint8, axis=0).reshape(8, 16, 3)
fig = plt.figure(); plt.imshow(img)
plt.show()
img = data_uint8.reshape(8, 16, 3)
fig = plt.figure(); plt.imshow(img)
plt.show()

L = kneighbors_graph(data_scaled_to01, 8, mode='distance', include_self=False)
ML = np.eye(128) - csgraph_laplacian(L, normed=True, return_diag=False, symmetrized=True)

# truth
V = rng.standard_normal((128, 1))
_, V = lobpcg(ML, V, Y=np.ones((128, 1)), maxiter=30)
labels = ( V > 0 ) * 1
print("truth:", labels.T)

# pred
V = rng.standard_normal((128, 1))
V = power_Fiedler(ML, V, maxit=100)
labels = ( V > 0 ) * 1
print("prediction:", labels.T)

# my prediction
r2, v2 = power_iter_for_second(ML, MAX_STEPS=100, verbose=False)
labels = (v2 > 0) * 1
print("results of my fn:", labels.T)


