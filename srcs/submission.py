from training import gradDescent
import numpy as np

def submit(mtx, mean, eigenvec):
    mtxflat = mtx.flatten()
    mtxflat = mtxflat - mean
    mtxflat = np.dot(np.matrix(eigenvec[0]), np.matrix(mtxflat))
    theta = np.random.rand(np.shape(eigenvec)[1])
    theta = gradDescent(eigenvec, np.matrix(mtxflat), np.matrix(theta).transpose(), 1e-16, 150)
    return theta

