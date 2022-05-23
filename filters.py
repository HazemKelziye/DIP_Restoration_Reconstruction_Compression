# import cv2
# import os

# import matplotlib.pyplot as plt
import numpy as np


# from scipy import stats


def gaussian(image):
    row, col, ch = image.shape
    mean = 50
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.array(noisy, dtype=np.uint8)
    return noisy

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
