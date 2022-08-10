import numpy as np


def cal_eigenvalue(A, v):
    # Rayleigh quotient:
    # lambda = (v* A v) / (v* v)
    # A: dxd v:dx1 Av:dx1
    Av = A.dot(v)
    #return np.dot(Av.T, v) #/ np.dot(v.T, v)
    return v.T.dot(Av)

def power_iter(M, MAX_STEPS=100):
    n, d = M.shape
    threshold = 1e-5 / n

    # initialize v randomly
    v = np.random.rand(d,)
    v = v / np.linalg.norm(v, ord=2)
    pre_diff = np.abs(v)

    # loop until converge
    for i in range(MAX_STEPS):
        pre_v = v
        print(f"=> step {i}: v = {v}")
        v = M.dot(v)
        v = v / np.linalg.norm(v, ord=2)
        diff = np.abs(v - pre_v)
        err = np.linalg.norm(diff-pre_diff, ord=np.inf)
        print("   err:", err)
        if err < threshold:
            ev_k = cal_eigenvalue(M, v)
            print(f"find solution at step {i}: ev = {ev_k}, v = {v}")
            return ev_k, v
        pre_diff = diff
    raise ValueError(f"Power iteration did NOT converge in {MAX_STEPS} steps")

def get_abs_max(v):
    r = 0
    for i in range(len(v)):
        if np.abs(v[i]) > r:
            r = v[i]
    return r

if __name__=="__main__":
    np.random.seed(22)
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

    r, v = power_iter(A1)
