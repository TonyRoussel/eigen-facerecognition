import numpy as np

def concatMatrix(mtxLst):
    mtx = np.array(())
    flatLst = []
    for m in mtxLst:
        flatLst.append(m.flatten())
    mtx = np.vstack(flatLst)
    return mtx.transpose()

def extractEigenvecOnVal(eigval, eigvec, threshold = 1):
    delIdx = np.where(eigval < threshold)[0]
    return np.delete(eigvec, delIdx, axis=1)

def reconstructVector(M, eigvec):
    eigvecT = eigvec.transpose()
    newmatrix = np.empty(np.shape(eigvecT))
    for idx, vec in enumerate(eigvecT):
        newvec = np.dot(M, vec.transpose())
        newmatrix[idx] = newvec
    return newmatrix.transpose()
