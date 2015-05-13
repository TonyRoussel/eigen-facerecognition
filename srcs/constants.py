training_path = "../faceset/faces/"
img_extension = ".pgm"


# Test data
import numpy as np

A = np.array([
    [1, 2],
    [6, 1],
])

B = np.array([
    [1, 6],
    [7, 3],
])

C = np.array([
    [ 9, 10],
    [1, 2],
])

D = np.array([
    [1, 2, 6],
    [3, 4, 8],
])

E = np.array([
    [5, 6, 2],
    [7, 8, 9],
])

F = np.array([
    [ 9, 10, 3],
    [11, 12, 7],
])

G = np.array([
    [1, 2, 6],
    [3, 4, 8],
    [3, 4, 8],
])

H = np.array([
    [5, 6, 2],
    [7, 8, 9],
    [7, 8, 9],
])

I = np.array([
    [ 9, 10, 3],
    [11, 12, 7],
    [11, 12, 7],
])

J = np.random.random_integers(0, 255, (3, 3))
K = np.random.random_integers(0, 255, (3, 3))
L = np.random.random_integers(0, 255, (3, 3))

mtxLst = [A, B, C]
# mtxLst = [D, E, F]
# mtxLst = [J, K, L]
# !Test data
