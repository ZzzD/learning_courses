### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import numpy as np
import sounddevice as sd

Fs = 11025


def g(x):
    return 1. / (1 + np.exp(-x))


def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))


def load_data():
    mix = np.loadtxt('mix.dat')
    return mix


def play(vec):
    sd.play(vec, Fs, blocking=True)


def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    mask = list(range(M))
    np.random.shuffle(mask)

    for alpha in anneal:
        for i in mask:
            W += alpha * ((1 - 2 * g(W.dot(X[i][:, np.newaxis]))).dot(X[i][np.newaxis, :]) +
                          np.linalg.inv(W.T))
    ###################################
    return W


def unmix(X, W):

    ######### Your code here ##########
    S = X.dot(W.T)
    ##################################
    return S


def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])


if __name__ == '__main__':
    main()
