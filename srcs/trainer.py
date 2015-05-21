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
    eigenval, eigenvec = qr(Mtld, 400)
#     eigenval, eigenvec = np.linalg.eig(Mtld)
    eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec, 1)
    eigenvec = trn.reconstructVector(M, eigenvec)
    print "eigenvec shape: ", np.shape(eigenvec)
    for i in range(n):
        img = M.transpose()[i]
        theta = np.ones(np.shape(eigenvec)[1])
        theta = trn.gradDescent(eigenvec, np.matrix(img), np.matrix(theta).transpose(), learRate, maxIteration)
        print "Descent terminated: ", i, " / ", n - 1  ######
        thetas.append(theta)
    return (Mmean, eigenvec, thetas)
