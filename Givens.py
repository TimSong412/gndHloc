import numpy as np
from numpy.linalg import norm


def genGivens(w: np.ndarray):
    G = np.zeros((3, 3))
    G[0, 0] = w[1]/norm(w[0:2])
    G[0, 1] = -w[0]/norm(w[0:2])
    G[0, 2] = 0
    G[1, 0] = w[0]*w[2]/(norm(w[0:2])*norm(w[0:3]))
    G[1, 1] = w[1]*w[2] / (norm(w[0:2])*norm(w[0:3]))
    G[1, 2] = -norm(w[0:2])/norm(w[0:3])
    G[2, 0] = w[0]/norm(w[0:3])
    G[2, 1] = w[1]/norm(w[0:3])
    G[2, 2] = w[2]/norm(w[0:3])
    return G


def main():
    p1 = np.array([1, 2, 3, 4], dtype=np.float32)
    p1norm = p1[0:3]/np.linalg.norm(p1[0:3])

    Hp1 = genGivens(p1[0:3])
    print(Hp1)
    print(Hp1@p1norm)


if __name__ == "__main__":
    main()
