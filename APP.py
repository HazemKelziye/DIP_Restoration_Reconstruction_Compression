import cv2 as cv
from filters import *

image = cv.imread("Nature.jpg")
image = cv.resize(image, (800, 600), interpolation=cv.INTER_LINEAR)
cv.imshow('image', image)

noisy = noise_salt_pepper(image, 0.01)
noisy = cv.resize(noisy, (800, 600), interpolation=cv.INTER_LINEAR)
cv.imshow('noisy', noisy)
cv.imwrite("noisy.jpeg", noisy)
cv.imshow('denoised', median_filter(noisy))

cv.waitKey(0)
