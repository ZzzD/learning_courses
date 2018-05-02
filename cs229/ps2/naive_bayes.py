import numpy as np


trainMatrix = np.load('trainMatrix.npy')
trainCategory = np.genfromtxt('spam_data/MATRIX.TRAIN_category.csv', delimiter=',').astype(np.int)

testMatrix = np.load('testMatrix.npy')
testCategory = np.genfromtxt('spam_data/MATRIX.TEST_category.csv', delimiter=',').astype(np.int)

tokens = np.genfromtxt('spam_data/TOKENS_LIST', dtype=np.str)[:, 1]
m_train, v = trainMatrix.shape
m_test = testCategory.shape[0]


def get_phi_positive():
    pos_samples = trainMatrix[trainCategory.astype(np.bool)]
    return (np.sum(pos_samples, axis=0) + 1) / (np.sum(pos_samples) + v)


def get_phi_negative():
    neg_samples = trainMatrix[~trainCategory.astype(np.bool)]
    return (np.sum(neg_samples, axis=0) + 1) / (np.sum(neg_samples) + v)


phi_pos = get_phi_positive()
phi_neg = get_phi_negative()


def predict(samples):
    pos_likelihood = np.sum(samples * np.log(phi_pos), axis=1)
    neg_likelihood = np.sum(samples * np.log(phi_neg), axis=1)
    return (pos_likelihood > neg_likelihood).astype(np.int)


################################
#     solution on ps2-3.a      #
################################


print('err on train: ', (~(trainCategory == predict(trainMatrix))).sum() / m_train)
print('err on test:  ', (~(testCategory == predict(testMatrix))).sum() / m_test)


################################
#     solution on ps2-3.b      #
################################

spam_indicative = tokens[np.argpartition(-np.log(phi_pos / phi_neg), 5)[:5]]
print(spam_indicative)