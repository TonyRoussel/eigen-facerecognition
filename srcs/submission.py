from training import gradDescent
import numpy as np

def submit(mtx, mean, eigenvec):
    print "mtx shape: ", np.shape(mtx) ###########
    print "mean: ", mean ###########
    print "eigenvec shape: ", np.shape(eigenvec) ###########
    mtxflat = mtx.flatten()
    mtxflat = mtxflat - mean[:, np.newaxis]
    theta = np.random.rand(np.shape(eigenvec)[1])
    theta = gradDescent(eigenvec, np.matrix(mtxflat), np.matrix(theta).transpose(), 1e-16, 150)
    return theta

