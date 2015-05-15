#coding:utf8

"""Run very simple tests for ksvd algorithm"""

import random
import ksvd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import orthogonal_mp
from skimage.draw import circle_perimeter, ellipse_perimeter, polygon, line

################# COLLECTION OF SHAPES #######################

cercle = np.zeros((10, 10), np.uint8)
cercle[circle_perimeter(4, 4, 3)] = 1

ellipse = np.zeros((10, 10), np.uint8)
ellipse[ellipse_perimeter(4, 4, 3, 5)] = 1

square = np.zeros((10, 10), np.uint8)
square[polygon(np.array([1, 4, 4, 1]), np.array([1, 1, 4, 4]))] = 1

dline = np.zeros((10, 10), np.uint8)
dline[line(1, 1, 8, 8)] = 1

shapes = [cercle, ellipse, square, dline]


################ GENERATING DICTIONARY ####################

D = np.zeros((100, 4))

for i in range(len(shapes)):
    D[:,i] = shapes[i].ravel()
    D[:,i] = D[:,i]/np.linalg.norm(D[:,i])

#################### RANDOM IMAGES #######################

def generate_img(i):
    """Generates combination of at most i shapes from D dictionnary"""
    output = np.zeros(100)
    for _ in range(i):
        output += D[:,random.randint(0, len(shapes)-1)]
    return output

X = []
for _ in range(30):
        X.append(generate_img(2))
for _ in range(30):
        X.append(generate_img(1))

X = np.array(X).T

####################### TEST KSVD #########################

model = ksvd.KSVD((100, 4), K = 2)
model.fit(X)

gamma = model.sparse_rep(X)

plt.subplot(2, 2, 1)
plt.imshow(model.D[:,0].reshape((10,10)), cmap = 'gray')
plt.subplot(2, 2, 2)
plt.imshow(model.D[:,1].reshape((10,10)), cmap = 'gray')
plt.subplot(2, 2, 3)
plt.imshow(model.D[:,2].reshape((10,10)), cmap = 'gray')
plt.subplot(2, 2, 4)
plt.imshow(model.D[:,3].reshape((10,10)), cmap = 'gray')
plt.show()

for i in range(5):
        plt.subplot(5, 2, i*2 + 1)
        plt.imshow(X[:,i].reshape((10,10)), cmap = 'gray')
        plt.subplot(5, 2, i*2 + 2)
        plt.imshow(model.D.dot(gamma[:,i]).reshape((10,10)), cmap = 'gray')
plt.show()
