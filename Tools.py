import cv2
import numpy as np


def loadimage(str):
    img = cv2.imread('img/' + str, 0)
    # cv2.imshow('image', img)
    # k = cv2.waitKey(0)
    # print img.shape

    res = cv2.resize(img, (28, 28))
    # cv2.imshow('image', res)
    # k = cv2.waitKey(0)

    dst = np.zeros(28 * 28, dtype=float)
    dst = cv2.normalize(res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.reshape(dst, 784)
    # cv2.calcHist(res, 0, None, )

def showimage(src):
    img = np.reshape(src, [28, 28])
    cv2.imshow('image', img)
    cv2.waitKey(500)
    cv2.destroyWindow('image')