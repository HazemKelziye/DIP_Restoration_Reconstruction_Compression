import cv2 as cv
from filters import *

image = cv.imread("Nature.jpg")
image = cv.resize(image, (800, 600), interpolation=cv.INTER_LINEAR)
cv.imshow('image', image)

noisy = gaussian(image)
noisy = cv.resize(noisy, (800, 600), interpolation=cv.INTER_LINEAR)
cv.imshow('noisy', noisy)

cv.waitKey(0)
