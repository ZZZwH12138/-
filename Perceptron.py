import numpy as np
import random
import matplotlib.pyplot as plt


def sign(v):
    if v > 0:
        return 1
    else:
        return 0


np.random.seed(12)
num_observations = 500

x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)
X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

# print(x1)
w = [0, 0]
w_0 = 0
r = 0.01
y = np.zeros(1000)
loss = 1
while loss > 0.005:
    i = random.randint(0, 999)
    x = X[i]
    d = Y[i]
    y[i] = sign(w[0] * x[0] + w[1] * x[1] + w_0)
    w[0] = w[0] + r * (d - y[i]) * x[0]
    w[1] = w[1] + r * (d - y[i]) * x[1]
    w_0 = w_0 + r * (d - y[i])

    loss_sum = 0
    for k in range(1000):
        loss_sum = loss_sum + abs(Y[k] - y[k])
    loss = loss_sum / 1000
    print('w[0]:', w[0], '|', 'w[1]:', w[1], '|', 'w_0:', w_0, '|', 'loss:', loss)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.9, cmap=plt.cm.bone, edgecolor='black')


xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
xy = np.c_[xx.ravel(), yy.ravel()]
Z = np.dot(xy, w)
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[-w_0], colors='b')

plt.show()
