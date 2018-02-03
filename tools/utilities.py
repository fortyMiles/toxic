import numpy as np


def softmax(A):
    A = np.array(A, dtype=float)
    A -= np.mean(A)
    return np.exp(A) / np.sum(np.exp(A))


if __name__ == '__main__':
    print(softmax([1, 2, 3]))