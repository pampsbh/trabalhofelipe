import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Pessoas/face0.png',0)
plt.hist(img.ravel(),256,[0,256])
histg = cv2.calcHist([img],[0],None,[256],[0,256])
plt.show()

#histg = cv2.calcHist([img],[0],None,[256],[0,256])