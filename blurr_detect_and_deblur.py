from imutils import paths
import argparse
import cv2
import sys
import numpy as np
import pylops
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2


def variance_of_laplacian(image):
	# calculates the Laplacian and returns the variance of the Laplacian(focus) of the image.
	# can be used to detect blur
	return cv2.Laplacian(image, cv2.CV_64F).var()


# smoothens the image
def smooth(image):
	kernel = np.ones((5,5),np.float32)/25
	#smooth = cv2.filter2d(image, -1, kernel)    // another way to smoothen the image
	gaussian_blur = cv2.GaussianBlur(image,(5,5),0) 
	cv2.imwrite("yo.png", gaussian_blur)


# deblurs the image
def deblur(image):
	pic = color.rgb2gray(image)
	psf = np.ones((5, 5)) / 25
	pic = conv2(pic, psf, 'same')
	pic += 0.1 * pic.std() * np.random.standard_normal(pic.shape)
	deconvolved, _ = restoration.unsupervised_wiener(pic, psf)
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)
	plt.gray()
	ax[0].imshow(pic, vmin=deconvolved.min(), vmax=deconvolved.max())
	ax[0].axis('off')
	ax[0].set_title('Data')
	ax[1].imshow(deconvolved)
	ax[1].axis('off')
	ax[1].set_title('Self tuned restoration')
	fig.tight_layout()
	plt.show()
	cv2.imwrite("yo.png", deconvolved)


def main():
	image  = sys.argv[1]
	image = cv2.imread(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	print(fm)
	deblur(image)
	if fm < 100.0:
			text = "Blurry"
			deblur(image)
	else:
		text = "Not Blurry"


if __name__ == '__main__':
	main()
