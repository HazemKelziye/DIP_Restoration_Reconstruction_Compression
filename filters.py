# import cv2
# import os
# import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import skimage

def gaussian(image):
    row, col, ch = image.shape
    mean = 0.2
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss = np.array(gauss, dtype=np.uint8)
    noisy = cv.add(image, gauss)
    return noisy

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    return image

# def salt_pepper(image):
#     prob = 0.01
#     threshold = 1 - prob
#     for i in range(0, image.shape[0]):
#         for j in range(0, image.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 image[i][j] = 0
#             elif rdn > thres:
#                 image[i][j] = 255
#             else:
#                 image[i][j] = image[i][j]
#     plt.hist(image, bins='auto')
#     plt.show()
#     return image
#
#     elif noise_type == "poisson".lower():
#     vals = len(np.unique(image))
#     vals = 2 ** np.ceil(np.log2(vals))
#     noisy = np.random.poisson(image * vals) / float(vals)
#     noisy = np.array(noisy, dtype=np.uint8)
#     plt.hist(noisy, bins=5)
#     plt.show()
#     return noisy
#
# elif noise_type == "speckle".lower():
# row, col = image.shape
# mean = 0
# var = 0.1
# gauss = np.random.normal(mean, math.sqrt(var), (row, col))
# noisy = image + image * gauss
# noisy = np.array(noisy, dtype=np.uint8)
# # plt.hist(noisy,bins='auto')
# # plt.show()
# return noisy
# elif noise_type == "rayleigh".lower():
# row, col = image.shape
# mean = 0
# var = 0.1
# sigma = var ** 0.5
# s = stats.mode(image, axis=None)
# print("s=", s[0])
# s = s[0]
# ray = np.random.rayleigh(scale=10, size=(row, col))
# # print(gauss.shape)
# # plt.hist(ray, bins='auto')
# # plt.show()
# ray = ray.reshape(row, col)
# noisy = image + ray
# noisy = np.array(noisy, dtype=np.uint8)
# return noisy
# elif noise_type == "Gamma".lower():
# row, col = image.shape
# gamma = np.random.gamma(shape=2, scale=10, size=(row, col))
# plt.hist(gamma, bins='auto')
# plt.show()
# image = image + gamma
# image = np.array(image, dtype=np.uint8)
# # plt.hist(image, bins='auto')
# # plt.show()
# return image
# elif noise_type == "Exponential".lower():
# row, col = image.shape
# exponen = np.random.exponential(scale=3, size=(row, col))
# plt.hist(exponen, bins='auto')
# plt.show()
# image = image + exponen
# image = np.array(image, dtype=np.uint8)
# plt.hist(image, bins='auto')
# plt.show()
# return image
