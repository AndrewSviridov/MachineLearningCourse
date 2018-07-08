import skimage.io
from skimage import img_as_float, img_as_ubyte
from skimage.measure import compare_psnr
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

image = img_as_float(skimage.io.imread('popugs.jpg'))

[x_size, y_size] = image.shape[0], image.shape[1]

X = []
median_img_list = []
mean_img_list = []

for x in xrange(x_size):
    for y in range(y_size):
        X.append(image[x, y])
        median_img_list.append(image[x, y])
        mean_img_list.append(image[x, y])

for k in xrange(2, 20):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=241)
    kmeans.fit(X)

    mean_color = np.zeros((kmeans.n_clusters, 3))
    median_color = np.zeros((kmeans.n_clusters, 3))

    for i in xrange(kmeans.n_clusters):
        indices = np.array([X[arg[0]] for arg in np.argwhere(kmeans.labels_ == i)])
        mean_color[i] = np.mean(indices[:, 0]), np.mean(indices[:, 1]), np.mean(indices[:, 2])
        median_color[i] = np.median(indices[:, 0]), np.median(indices[:, 1]), np.median(indices[:, 2])

        for arg in np.argwhere(kmeans.labels_ == i):
            mean_img_list[arg[0]] = tuple(img_as_ubyte(mean_color[i]))
            median_img_list[arg[0]] = tuple(img_as_ubyte(median_color[i]))

    mean_img = Image.new("RGB", (y_size, x_size), (0, 0, 0, 0))
    mean_img.putdata(mean_img_list)

    median_img = Image.new("RGB", (y_size, x_size), (0, 0, 0, 0))
    median_img.putdata(median_img_list)

    # Image._show(mean_img)
    # Image._show(median_img)

    psnr_mean = compare_psnr(image, img_as_float(mean_img))
    psnr_median = compare_psnr(image, img_as_float(median_img))

    print("Mean: k = {}, PSNR = {}".format(kmeans.n_clusters, psnr_mean))
    print("Median: k = {}, PSNR = {}".format(kmeans.n_clusters, psnr_median))