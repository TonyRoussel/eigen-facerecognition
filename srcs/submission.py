from training import gradDescent
from constants import learRate
from constants import maxIteration
import numpy as np
import matplotlib.pyplot as plt


def submit(mtx, mean, eigenvec):
    mtxflat = mtx.flatten()
    mtxflat = np.vstack(list(mtxflat))
    mtxflat -= mean[:, np.newaxis]
    mtxflat = mtxflat.transpose()[0]
    
#     eigenvecOne = np.insert(eigenvec, 0, np.ones(np.shape(eigenvec)[0]), axis=1)
    theta = np.ones(np.shape(eigenvec)[1])
    theta = gradDescent(eigenvec, np.matrix(mtxflat), np.matrix(theta).transpose(), learRate, maxIteration)
    # plt.imshow(np.reshape(mtxflat, (112, 92)))
    # plt.gray()
    # plt.show()
    # plt.imshow(np.reshape(np.dot(eigenvec, theta), (112, 92)))
    # plt.gray()
    # plt.show()
    return theta

def compare(thetaSubmit, thetas):
    minIdx = 0
    minVal = np.absolute(np.sum(thetas[0] - thetaSubmit))
    for idx, theta in enumerate(thetas):
        val = np.absolute(np.sum(np.absolute(theta - thetaSubmit)))
        if val < minVal:
            minIdx = idx
            minVal = val
    return minIdx

def compareAvgGap(thetaSubmit, thetas):
    minIdx = 0
    minVal = (np.absolute(thetas[0] - thetaSubmit)).mean()
    for idx, theta in enumerate(thetas):
        val = (np.absolute(theta - thetaSubmit)).mean()
        if val < minVal:
            minIdx = idx
            minVal = val
    return minIdx
