import numpy as np
import cv2 as cv
import random


def noise_gaussian(image, mean, var):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = np.array(gauss, dtype=np.uint8)
    noisy = cv.add(image, gauss)
    return noisy

def noise_salt_pepper(image, prob):

    threshold = 1 - prob

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            # return a value between [0, 1]
            rdn = random.random()

            if rdn < prob:
                image[i][j] = 0

            elif rdn > threshold:
                image[i][j] = 255

            else:
                image[i][j] = image[i][j]
    return image

def noise_rayleigh(image, mean):
    row, col, ch = image.shape
    ray = np.random.rayleigh(scale=mean, size=(row, col, ch))
    ray = np.array(ray, dtype=np.uint8)
    noisy = cv.add(image, ray)
    return noisy

def noise_gamma(image, mean, variance):
    row, col, ch = image.shape
    gamma = np.random.gamma(shape=mean, scale=variance, size=(row, col, ch))
    gamma = np.array(gamma, dtype=np.uint8)
    image = cv.add(image, gamma)
    return image

def noise_exponential(image, mean):
    row, col, ch = image.shape
    expo = np.random.exponential(scale=mean, size=(row, col, ch))
    expo = np.array(expo, dtype=np.uint8)
    image = cv.add(image, expo)
    return image

def noise_uniform(image, a, b):
    row, col, ch = image.shape
    uni = np.random.uniform(low=a, high=b, size=(row, col, ch))
    uni = np.array(uni, dtype=np.uint8)
    image = cv.add(image, uni)
    return image