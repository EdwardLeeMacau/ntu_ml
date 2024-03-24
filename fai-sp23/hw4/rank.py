import numpy as np
from numpy import linalg as LA
from scipy.linalg import null_space

def main():
    # page rank
    v = np.array([1/3, 1/3, 1/3])
    P = np.array([
        [0, 1, 1/2],
        [0, 0, 1/2],
        [1, 0, 0],
    ])

    # for i in range(100):
    #     v = P @ v
    #     v = v / np.sum(v)
    #     print(f'{v=}')

    print(f'{LA.eig(P)=}')
    print(f'{LA.inv(P)=}')
    print(f'{np.eye(3) - P=}')

    print(f'{null_space(np.eye(3) - P)=}')

if __name__ == "__main__":
    main()