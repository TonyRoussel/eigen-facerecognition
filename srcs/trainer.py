import numpy as np
import training as trn
from qr import qr
from constants import learRate
from constants import maxIteration
import matplotlib.pyplot as plt

def train(mtxLst):
    thetas = list()
    M = trn.concatMatrix(mtxLst)
    Mmean = M.mean(axis=1)
    M -= Mmean[:, np.newaxis]
    Mtld = np.dot(M.transpose(), M)
    n = np.shape(Mtld)[1]
    eigenval, eigenvec = qr(Mtld, 100)
#     eigenval, eigenvec = np.linalg.eig(Mtld)
    # eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec, 1)
    eigenvec = trn.reconstructVector(M, eigenvec)
    print "eigenvec shape: ", np.shape(eigenvec)
    thetas = np.dot(M.transpose(), eigenvec)
    return (Mmean, eigenvec, thetas)
