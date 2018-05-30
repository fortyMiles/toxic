import numpy as np


def softmax(A):
    A = np.array(A, dtype=float)
    A -= np.mean(A)
    return np.exp(A) / np.sum(np.exp(A))


def split(string : str):
    string = string.replace(',', ' ')
    return string.split()


assert split('I am your father') == ['I', 'am', 'your', 'father']
assert split('I, am , your, father') == ['I', 'am', 'your', 'father']

if __name__ == '__main__':
    print(softmax([1, 2, 3]))

