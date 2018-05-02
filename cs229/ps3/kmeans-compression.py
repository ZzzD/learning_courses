import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

src_img = np.array(Image.open('mandrill-large.tiff'))
Image.open('mandrill-large.tiff').show()
h, w, channel = src_img.shape
k = 16


def init_cluster(k):
    x = np.random.randint(h, size=k)
    y = np.random.randint(h, size=k)
    return src_img[x, y, :]


def cluster(c, k):
    dist = np.zeros((h, w, k))
    for i in range(k):
        dist[:, :, i] = np.linalg.norm(src_img - c[i], axis=2)
    return np.argmin(dist, axis=2)


def adj_centroid(dist, k):
    c = np.zeros((k, channel))
    for i in range(k):
        c[i] = np.mean(src_img[dist==i], axis=0)
    return c


def compression(dist, c):
    tar_img = np.zeros_like(src_img)
    for i in range(h):
        for j in range(w):
            tar_img[i, j, :] = c[dist[i, j]]
    plt.imshow(tar_img)
    plt.show()


def kmeans():
    centroid = init_cluster(k)
    iter = 0
    cvg = False
    while not cvg:
        old_centroid = centroid
        d = cluster(centroid, k)
        centroid = adj_centroid(d, k)
        if np.array_equal(centroid, old_centroid):
            cvg = True
            print('cvg: ', iter)
            compression(d, centroid)
        iter += 1

kmeans()
