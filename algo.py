import numpy as np
import time
from sklearn.cluster import KMeans
from utils import k_means, cluster_qr, check_random_state, sign
from power_iteration import power_iter
from scipy.sparse.linalg import lobpcg

def cal_eigenvalue(A, v):
    # Rayleigh quotient:
    # lambda = (v* A v) / (v* v)
    # A: dxd v:dx1 Av:dx1
    Av = A.dot(v)
    #return np.dot(Av.T, v) #/ np.dot(v.T, v)
    return v.T.dot(Av)

def labeling(
        maps,
        mode="kmeans",
        n_clusters=8,
        random_state=22,
        n_init=10,
        verbose=False,
        ):
    print(f"Computing label assignment using {mode}")
    if mode == "kmeans":
        _, labels, _ = k_means(
            maps, n_clusters, random_state=random_state, n_init=n_init, verbose=verbose
        )
    elif mode == "cluster_qr":
        labels = cluster_qr(maps)
    elif mode == "sign":
        labels = sign(maps)
    else:
        raise ValueError(
            "The 'assign_labels' parameter should be "
            "'kmeans' or 'sign', or 'cluster_qr', "
            f"but {assign_labels!r} was given"
        )
    return labels

def make_laplacian(A):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('csgraph must be a square matrix or array')
    D = np.diag(A.sum(0))
    L = D - A
    return L, D

def make_normalized_ajacency(A):
    D = np.diag(A.sum(0))
    n,d = A.shape
    W = np.zeros((n,d))
    for i in range(n):
        for j in range(d):
            if i!=j and A[i][j]==1:
                W[i][j] = 1 / np.sqrt(D[i][i]*D[j][j])
    return W

def make_symmetrically_normalized_Laplacian(A):
    """
    A: ajacency matrix
    output: symmetrically_normalized_Laplacian L
    """
    n,d = A.shape
    if n!=d:
        raise ValueError("A should be symmentric")
    D = np.diag(A.sum(0))
    sml = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j and D[i][j]!=0:
                sml[i][j] = 1
            elif i!=j and A[i][j]==1:
                sml[i][j] = - 1 / np.sqrt(D[i][i]*D[j][j])
            else:
                sml[i][j] = 0
    return sml

def spectral_clustering(
        A, 
        n_clusters=8,
        n_components=None,
        norm_laplacian=True,
        assign_labels="sign",
        n_init=10,
        random_state=22,
        verbose=False
        ):
    if isinstance(A, np.matrix):
        raise TypeError(
            "spectral_clustering does not support passing in affinity as an "
            "np.matrix. Please convert to a numpy array with np.asarray. For "
            "more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html", 
        )

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    tt1 = time.perf_counter()
    # pre-norm A
    # approach1:
    # D = np.diag(A.sum(0))
    # W = np.linalg.inv(np.sqrt(D)) @ A @ np.linalg.inv(np.sqrt(D))
    # approach2:
    # W = np.linalg.inv(D) @ A
    # approach3:
    # W = make_normalized_ajacency(A)
    # apply l1-norm for each row
    # W = W / np.linalg.norm(W, axis=1, ord=1)[:,None]
    # approach4:
    W = np.eye(A.shape[0]) - make_symmetrically_normalized_Laplacian(A)
    tt2 = time.perf_counter()
    print("pre-normalized A:", W)

    # cal 2nd smallest eigenvalue and its cooresponding eigenvector
    t0 = time.perf_counter()
    rng = np.random.default_rng()
    x = rng.standard_normal((A.shape[0], 1))
    v2 = power_Fiedler(W, x, maxit=100)
    # r2, v2 = power_iter_for_second(W)

    t1 = time.perf_counter()
    labels = labeling(
            v2, 
            mode=assign_labels, 
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init=10,
            verbose=False)
    t2 = time.perf_counter()
    print("prepare time cost:", (tt2-tt1))
    print("power iteration time cost:", (t1-t0))
    print("labeling time cost:", (t2-t1))
    print("predicted labels:", labels)

    # get true labels
    V = rng.standard_normal((A.shape[0], 1))
    _, V = lobpcg(W, V, Y=np.ones((A.shape[0], 1)), maxiter=30)
    true_labels = ( V > 0 ) * 1
    print("true labels:", true_labels.T)
    return labels


def power_iter_for_second(M, MAX_STEPS=100, verbose=True):
    # M is the pre-normalized adjacency matrix,
    # row_norm( inv(sqrt(D)) @ A @ inv(sqrt(D)) ),
    # which has r1=1 and v1=c[1,...,1].T
    n, d = M.shape
    threshold = 1e-5 / n

    # initialize v randomly
    v = np.random.rand(d,1)
    v = v - v.mean()
    # v = v / np.linalg.norm(v, ord=2)
    pre_diff = np.abs(v)

    # loop until converge
    for i in range(MAX_STEPS):
        pre_v = v
        # orthogonal to 1st eigenvector
        v = M @ v 
        v = v - v.mean()
        # v = v / np.linalg.norm(v, ord=2)
        diff = np.abs(v - pre_v)
        err = np.linalg.norm(diff-pre_diff, ord=np.inf)
        if verbose:
            print(f"=> step {i}: v = {v}, err = {err}")
        if err < threshold:
            ev_k = cal_eigenvalue(M, v) 
            if verbose:
                print(f"Find solution at step {i}:\neigenvalue = {ev_k},\neigenvector = {v}")
            return ev_k, v 
        pre_diff = diff
    print(f"Warnning: not reach the request tolerance {threshold} at {MAX_STEPS} steps")
    ev_k = cal_eigenvalue(M, v)
    return ev_k, v

def power_Fiedler(A, x, maxit=100):
    for _ in range(maxit):
        x = A @ x
        norm_factor = np.sum(x) / A.shape[0]
        x = x - norm_factor
    return x


if __name__=="__main__":
    # np.random.seed(22)
    # tiny graph 6x6
    A1 = np.array([[0,1,1,0,0,1],
                   [1,0,1,0,0,0],
                   [1,1,0,1,0,0],
                   [0,0,1,0,1,1],
                   [0,0,0,1,0,1],
                   [1,0,0,1,1,0]])
    # small graph 12x12
    A2 = np.array([[0,1,1,0,0,0,0,0,0,0,0,0],
                   [1,0,1,0,0,0,0,0,0,0,0,0],
                   [1,1,0,1,0,0,0,0,0,0,0,0],
                   [0,0,1,0,1,1,1,0,0,0,0,0],
                   [0,0,0,1,0,1,1,0,0,0,0,0],
                   [0,0,0,1,1,0,1,0,1,0,0,0],
                   [0,0,0,1,1,1,0,1,0,0,0,0],
                   [0,0,0,0,0,0,1,0,1,0,1,1],
                   [0,0,0,0,0,1,0,1,0,1,1,0],
                   [0,0,0,0,0,0,0,0,1,0,1,0],
                   [0,0,0,0,0,0,0,1,1,1,0,1],
                   [0,0,0,0,0,0,0,1,0,0,1,0]])

    labels = spectral_clustering(
        A2,
        n_clusters=2,
        n_components=2,
        norm_laplacian=True,
        assign_labels="sign",
        n_init=10,
        random_state=42,
        verbose=False
        )

