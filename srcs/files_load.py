from scipy import misc
import constants as cst
import os
import time

usleep = lambda x: time.sleep(x/1000000.0)

def loadmatrixs(path):
    matrixs = []
    sze = len(os.listdir(path))
    for i, filename in enumerate(os.listdir(path)):
        if not filename.endswith(cst.img_extension):
            continue
        img = misc.imread(path + filename)
        matrixs.append((filename, img))
        usleep(250)
    return matrixs
