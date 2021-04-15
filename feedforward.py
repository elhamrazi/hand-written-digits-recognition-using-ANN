import numpy as np
import math
from read_dataset import *


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def guess_the_number(mat):
    max = -10000
    index = 0
    for i in range(len(mat)):
        if mat[i] > max:
            max = mat[i]
            index = i
    return index


# create the matrix of weights randomly and the biases.
w0 = np.random.randn(16, 784)
w1 = np.random.randn(16, 16)
w2 = np.random.randn(10, 16)
b = np.zeros((16, 1))
b3 = np.zeros((10, 1))

right_guesses = 0

for i in range(100):
    a1 = (w0 @ train_set[i][0]) + b
    a11 = np.asarray([sigmoid(i) for i in a1]).reshape((16, 1))
    a2 = (w1 @ a11) + b
    a22 = np.asarray([sigmoid(i) for i in a2]).reshape((16, 1))
    a3 = (w2 @ a22) + b3
    a33 = np.asarray([sigmoid(i) for i in a3]).reshape((10, 1))
    print(guess_the_number(a3), guess_the_number(train_set[i][1]))
    if guess_the_number(a33) == guess_the_number(train_set[i][1]):
        right_guesses += 1

print("the accuracy of this model is:", right_guesses / 100)
# Plotting an image
print(guess_the_number(train_set[1][1]))
print(train_set[1][1])
show_image(train_set[1][0])
plt.show()







