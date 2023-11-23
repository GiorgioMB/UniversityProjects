#%%
import numpy as np
import cv2
def mse(imageA, imageB):
	if imageA.shape != imageB.shape:
		imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

reference = cv2.imread("reference.jpeg")
error = cv2.imread("MOSHED 21.jpg")
mse(reference, error)

# %%