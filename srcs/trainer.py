import numpy as np
import training as trn
from qr import qr
from constants import learRate
import matplotlib.pyplot as plt

def train(mtxLst):
    thetas = list()
    M = trn.concatMatrix(mtxLst)
    Mmean = M.mean(axis=1)
    M -= Mmean[:, np.newaxis]
    Mtld = np.dot(M.transpose(), M)
    n = np.shape(Mtld)[1]
    eigenval, eigenvec = qr(Mtld, 20000)
    eigenvec = trn.extractEigenvecOnVal(eigenval, eigenvec)
    eigenvec = trn.reconstructVector(M, eigenvec)
    for i in range(np.shape(eigenvec.transpose())[0]):
        plt.imshow(np.reshape(eigenvec.transpose()[i], np.shape(mtxLst[0])))
        plt.gray()
        plt.show()
    print "eigenvec shape: ", np.shape(eigenvec)
    for i in range(n):
        img = M.transpose()[i]
        theta = np.ones(np.shape(eigenvec)[1])
        theta = trn.gradDescent(eigenvec, np.matrix(img), np.matrix(theta).transpose(), learRate)
        print "Descent terminated: ", i, " / ", n - 1  ######
        thetas.append(theta)
    # plt.imshow(np.reshape(np.dot(eigenvec, thetas[0]), np.shape(mtxLst[0])))
    # plt.gray()
    # plt.show()
    return (Mmean, eigenvec, thetas)
