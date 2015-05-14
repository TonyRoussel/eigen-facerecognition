from scipy import misc

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
    [1, 2],
    [3, 4],
])

E = np.array([
    [5, 6],
    [7, 8],
])

F = np.array([
    [ 9, 10],
    [11, 12],
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

J = np.random.random_integers(0, 255, (100, 100))
K = np.random.random_integers(0, 255, (100, 100))
L = np.random.random_integers(0, 255, (100, 100))
M = np.random.random_integers(0, 255, (100, 100))
N = np.random.random_integers(0, 255, (100, 100))
O = np.random.random_integers(0, 255, (100, 100))
P = np.random.random_integers(0, 255, (100, 100))

meryl01 = misc.imread('../faceset/faces/sample_train/Meryl_Streep_0001.pgm')
meryl02 = misc.imread('../faceset/faces/sample_train/Meryl_Streep_0002.pgm')

zinedine01 = misc.imread('../faceset/faces/sample_train/Zinedine_Zidane_0001.pgm')
zinedine02 = misc.imread('../faceset/faces/sample_train/Zinedine_Zidane_0002.pgm')

# mtxLst = [A, B, C]
# mtxLst = [D, E, F]
# mtxLst = [J, K, L]
# mtxLst = [J, K, L, M, N, O, P]
mtxLst = [meryl01, meryl02, zinedine01, zinedine02]
# !Test data
