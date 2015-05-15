from training import gradDescent
from constants import learRate
import numpy as np

def submit(mtx, mean, eigenvec):
    mtxflat = mtx.flatten()
    mtxflat = mtxflat[:, np.newaxis] - mean[:, np.newaxis]
#     eigenvecOne = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    theta = np.random.rand(np.shape(eigenvec)[1])
    theta = gradDescent(eigenvec, np.matrix(mtxflat).transpose(), np.matrix(theta).transpose(), learRate)
    return theta

def compare(thetaSubmit, thetas):
    minIdx = 0
    minVal = np.absolute(np.sum(thetas[0] - thetaSubmit))
    for idx, theta in enumerate(thetas):
        val = np.absolute(np.sum(theta - thetaSubmit))
        if val < minVal:
            minIdx = idx
            minVal = val
    return minIdx
