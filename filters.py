import numpy as np
import cv2 as cv
import random

#Degradation (adding noise)
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

#Filtering (Denoising)
def median_filter(image, kernel_size=5):
    denoised = cv.medianBlur(image,kernel_size)
    return denoised

def bilateral_filter(image, d=15, siamaColor=75, sigamSpace=75):
    #d is the Diameter of each pixel neighborhood that is used during filtering.

    denoised = cv.bilateralFilter(image, d, siamaColor, sigamSpace)

    return denoised

def max_min_filter(img, K_size):
    '''Min filter used to find the darkest points in an image'''
    height, width = img.shape
    pad = K_size // 2
    out_img = img.copy()
    pad_img = np.zeros((height + pad*2, width + pad*2), dtype=np.uint8)
    pad_img[pad: pad+height, pad: pad+width] = img.copy()
    for y in range(height):
        for x in range(width):
            out_img[y,x] = np.max(pad_img[y:y+K_size, x:x+K_size]) - np.min(pad_img[y:y+K_size, x:x+K_size])
    return out_img

def max_min_filter(img, K_size):
    height, width = img.shape
    pad = K_size // 2
    out_img = img.copy()
    pad_img = np.zeros((height + pad*2, width + pad*2), dtype=np.uint8)
    pad_img[pad: pad+height, pad: pad+width] = img.copy()
    for y in range(height):
        for x in range(width):
            out_img[y,x] = np.max(pad_img[y:y+K_size, x:x+K_size]) - np.min(pad_img[y:y+K_size, x:x+K_size])
    return out_img
