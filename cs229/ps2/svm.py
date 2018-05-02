import numpy as np


trainMatrix = np.load('trainMatrix.npy')
trainMatrix = (trainMatrix != 0.).astype(np.int)

trainCategory = np.genfromtxt('spam_data/MATRIX.TRAIN_category.csv', delimiter=',').astype(np.int)

testMatrix = np.load('testMatrix.npy')
testMatrix = (testMatrix != 0.).astype(np.int)

testCategory = np.genfromtxt('spam_data/MATRIX.TEST_category.csv', delimiter=',').astype(np.int)

trainCategory = 2 * trainCategory - 1
testCategory = 2 * testCategory - 1

tokens = np.genfromtxt('spam_data/TOKENS_LIST', dtype=np.str)[:, 1]
m_train, v = trainMatrix.shape
m_test = testCategory.shape[0]

lambd = 1 / (64 * m_train)
tau = 8
K = np.load('kernel_matrix.npy')
K_test = np.load('test_kernel_matrix.npy')



def gaussian_kernel_matrix(x):
    K = np.zeros((x.shape[0], m_train))
    for i in range(x.shape[0]):
        mi = np.sum((trainMatrix - x[i]) ** 2, axis=1)
        K[i, :] = np.exp(-1 / (2 * (tau ** 2)) * mi)
        if not i % 100:
            print(i)
    return K

# np.save('kernel_matrix', K)


def cost_function(alpha):
    samples_err = np.mean(1 - trainCategory * K.T.dot(alpha), axis=0)
    reg_err = 0.5 * lambd * alpha[:, np.newaxis].T.dot(K).dot(alpha[:, np.newaxis])

    return samples_err + reg_err.item()


def step(alpha, t):
    ind = np.random.randint(m_train)
    dalpha = -(trainCategory[ind] * K[ind].dot(alpha) < 1).astype(np.int)\
             * trainCategory[ind] * K[ind]
    return alpha - dalpha / (t + 1) ** .5


def predict(alpha):
    # print(alpha.shape)
    return (testCategory * K_test.dot(alpha) > 0).astype(np.int) * 2 -1


def train():
    alpha = np.zeros((m_train,))
    avg_alpha = np.zeros_like(alpha)
    for t in range(40 * m_train):
        alpha = step(alpha, t)
        avg_alpha += alpha
    return avg_alpha


avg_alpha = train()
pred = predict(avg_alpha)
err = 100. - (pred == testCategory).sum() / m_test * 100
print('err: {}%'.format(err))



