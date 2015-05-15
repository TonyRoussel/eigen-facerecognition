import numpy as np
import training as trn
from qr import qr
from constants import learRate

def train(mtxLst):
    thetas = list()
    M = trn.concatMatrix(mtxLst)
    Mmean = M.mean(axis=1)
    M -= Mmean[:, np.newaxis]
    Mtld = np.dot(M.transpose(), M)
    n = np.shape(Mtld)[1]
    eigenval, eigenvec = qr(Mtld, 1000)
    eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec)
    eigenvec = trn.reconstructVector(M, eigenvec)
#     eigenvecOne = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    print "eigenvec shape: ", np.shape(eigenvec)
    for i in range(n):
        img = M.transpose()[i]
        theta = np.random.rand(np.shape(eigenvec)[1])
        theta = trn.gradDescent(eigenvec, np.matrix(img), np.matrix(theta).transpose(), learRate)
        print "Descent terminated: ", i, " / ", n - 1  ######
        thetas.append(theta)
    return (Mmean, eigenvec, thetas)
