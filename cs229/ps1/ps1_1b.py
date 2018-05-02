import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.genfromtxt('./logistic_x.txt')
m, n = x.shape

# add bias columns to x
x = np.hstack([np.ones((m, 1)), x])

y = np.genfromtxt('./logistic_y.txt')

def h(xi, yi, theta):
    return 1 / (1 + np.exp(-np.inner(xi, theta) * yi))

def dtheta(theta):
    dt = np.zeros((n + 1, ))
    for i in range(m):
        hi = h(x[i, :], -y[i], theta)
        dt += -1 / m * hi * y[i] * x[i, :]
    return dt

def ddtheta(theta):
    hess = np.zeros((n + 1, n + 1))
    for i in range(m):
        hi = h(x[i, :], -y[i], theta)
        hess += 1 / m * hi * (1 - hi) * np.outer(x[i, :], x[i, :])
    return hess


# theta = np.zeros((n + 1, ))
theta = np.array([-1.50983811, 0.43509696, 0.62161752])
for i in range(15):
    theta -= np.linalg.inv(ddtheta(theta)).dot(dtheta(theta))

print(theta)
# df = pd.DataFrame({'x': x[:, 1], 'y': x[:, 2]})
# fig, ax = plt.subplots()
#
# ax.scatter(df.x, df.y, c=np.sign(df.y), cmap="bwr")
#
#
#
# # plt.plot(xx, yy)
#
#
# plt.show()
