import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('quasar_train.csv', delimiter=',')

lambd = data[0, :]
m = lambd.shape[0]

train_set = data[1:, :]
test_set = np.genfromtxt('quasar_test.csv', delimiter=',')[1:, :]
lambd_add_bias = np.vstack(
    [np.ones(lambd.shape), lambd]).T
first_sample_x = train_set[0, :]


def weight_matrix(index, tau):
    return np.diag(np.exp(-np.square(lambd - lambd[index]) / (2 * tau ** 2)))


def local_weight_LR(tau):
    y_hat = np.zeros((m,))
    for i in range(m):
        Wi = weight_matrix(i, tau)
        theta_i = np.linalg.inv(lambd_add_bias.T.dot(Wi).dot(lambd_add_bias)) \
            .dot(np.dot(lambd_add_bias.T.dot(Wi), first_sample_x))
        y_hat[i] = theta_i[0] + lambd[i] * theta_i[1]
    return y_hat

################################
# unweighted linear regression #
################################


# theta = np.linalg.inv(y_add_bias.T.dot(y_add_bias))\
#                     .dot(np.dot(y_add_bias.T, first_sample_x))
#
# fig_x = np.arange(y.min(), y.max(), 1)
# fig_y = fig_x * theta[1] + theta[0]
#
# plt.plot(fig_x, fig_y, color='r')
#
# plt.scatter(y, first_sample_x, color='black', s=20, marker='+')
# plt.show()

################################
#  weighted linear regression  #
################################
for tau in [1, 5, 10, 100, 1000]:
    y_hat = local_weight_LR(tau)
    plt.plot(lambd, y_hat)

# fig_x = np.arange(y.min(), y.max() + 1, 1)


plt.scatter(lambd, first_sample_x, color='black', s=20, marker='+')
plt.show()


