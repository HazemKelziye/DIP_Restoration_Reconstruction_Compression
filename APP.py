import cv2 as cv
from filters import *



image = cv.imread("Lenna.png")
cv.imshow('image', image)

noisy = gaussian(image)
cv.imshow('noisy', noisy)

cv.waitKey(0)
