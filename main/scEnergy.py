#PURPOSE:
#Perform the seam carving resizing using different types of algorithms for the energy map calculation (e.g. Sobel, Prewitt, Laplacian, etc.)
#
#USAGE: (terminal)
#python3 scEnergy.py -in <image filename (in /images/ folder)> -out <output filename> -scale <downsizing scale> -seam <seam orientation, v for vertical h for horizontal>
#
#EXAMPLE: (terminal)
#python3 scEnergy.py -in image.jpg -out result.jpg -scale 0.5 -seam h -energy s


import sys
import os
import cv2
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from numba import jit
from scipy.ndimage.filters import convolve
from pathlib import Path as path

#This is to ignore NumbaWarnings and NumbaDeprecationWarnings issued by @jit
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#defines the new image shape based on scale provided
def resize(img, scale):
	#m = rows
	#n = columns
	m,n, _ = img.shape
	new_n = int(scale * n)

	for i in trange(n - new_n):
		img = seamCarving(img)

	return img 

@jit
#remove the seam (carving)
def seamCarving(img):
	#m = rows
	#n = columns
	m, n, _ = img.shape

	M, backtrack = findSeam(img)

	#creates a mask with value True in all positions
	mask = np.ones((m,n), dtype=np.bool)

	#finds the position of the smalletst element in the last row of M
	j = np.argmin(M[-1])
	#from bottom-up
	for i in reversed(range(m)):
		#marks the pixel for deletion
		mask[i,j] = False
		#gets the column position from the backtrack matrix
		j = backtrack[i,j]

	#converts the mask to 3D since the image has 3 channels
	mask = np.stack([mask] * 3, axis=2)

	#deletes the flagged pixels and resize the image to the new dimension
	img = img[mask].reshape((m, n - 1, 3))
	return img

@jit
#finds the minimal energy path (seam)
def findSeam(img):
	#m = rows
	#n = columns
	m,n, _ = img.shape

	#calculates the energy of each pixel using edge detection algorithms. e.g. Sobel, Prewitt, etc.
	energy_map = calculateEnergy(img)

	#the energy map is copied into M
	M = energy_map.copy()

	#creates the backtrack to find the list of pixels present in the found seam
	#backtrack is a matrix of zeroes with the same dimensions as the image/energy map/M
	backtrack = np.zeros_like(M, dtype=np.int)

	for i in range(1,m):
		for j in range(0,n):
			#if we are in the first column (the one more to the left)
			if j == 0:
				#index contains the minimal between M(i-1,j) and M(i-1,j+2) (pq nao j+1???)
				index = np.argmin(M[i - 1, j:j + 2]) #trocar por -1 e ver se tem alguma diferenca
				backtrack[i,j] = index + j 
				minimal_energy = M[i - 1, index + j]
			#if we are in the other columns
			else:
				#index contains the minimal between M(i-1,j-1), M(i-1,j) and M(i-1,j+2) (de novo, pq nao j+1???)
				index = np.argmin(M[i - 1, j - 1:j + 2]) #trocar por -1 também e ver se tem alguma diferenca
				backtrack[i,j] = index + j - 1
				minimal_energy = M[i-1, index+j-1]

			M[i,j] += minimal_energy

	return M, backtrack

@jit
#calculates the energy map using Sobel mask to find the seams to be removed
def calculateEnergy(img):
	if ENERGY_ALGORITHM == 's':
		energy_map = sobel(img)
	elif ENERGY_ALGORITHM == 'p':
		energy_map = prewitt(img)
	elif ENERGY_ALGORITHM == 'l':
		energy_map = laplacian(img)
	elif ENERGY_ALGORITHM == 'r':
		energy_map = roberts(img)
	else:
		energy_map = sobel(img)

	return energy_map

def sobel(img):
	kernelSy = np.array([
	    [1.0, 2.0, 1.0],
	    [0.0, 0.0, 0.0],
	    [-1.0, -2.0, -1.0],
	])
	# This converts it from a 2D filter to a 3D filter, replicating the same
	# filter for each channel: R, G, B
	kernelSy = np.stack([kernelSy] * 3, axis=2)

	kernelSx = np.array([
	    [1.0, 0.0, -1.0],
	    [2.0, 0.0, -2.0],
	    [1.0, 0.0, -1.0],
	])
	# This converts it from a 2D filter to a 3D filter, replicating the same
	# filter for each channel: R, G, B
	kernelSx = np.stack([kernelSx] * 3, axis=2)

	img = img.astype('float32')
	sobel = np.absolute(convolve(img, kernelSx)) + np.absolute(convolve(img, kernelSy))

	# We sum the energies in the red, green, and blue channels
	energy_map = sobel.sum(axis=2)

	#Saves the first energy map calculated (before any seam removed)
	global firstCalculation
	if firstCalculation == True:
		SOBEL_EDGE_PATH = EDGE_DETECTION_PATH + "sobel.jpg"
		cv2.imwrite(SOBEL_EDGE_PATH, np.rot90(sobel, 3, (0, 1)))

		SOBEL_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_sobel.jpg"
		cv2.imwrite(SOBEL_ENERGY_PATH, np.rot90(energy_map, 3, (0, 1)))
		firstCalculation = False

	return energy_map

def prewitt(img):
	kernelPx = np.array([
	    [1.0, 0.0, -1.0],
	    [1.0, 0.0, -1.0],
	    [1.0, 0.0, -1.0],
	])
	kernelPx = np.stack([kernelPx] * 3, axis=2)

	kernelPy = np.array([
	    [1.0, 1.0, 1.0],
	    [0.0, 0.0, 0.0],
	    [-1.0, -1.0, -1.0],
	])
	kernelPy = np.stack([kernelPy] * 3, axis=2)

	img = img.astype('float32')
	prewitt = np.absolute(convolve(img, kernelPx)) + np.absolute(convolve(img, kernelPy))

	energy_map = prewitt.sum(axis=2)

	#Saves the first energy map calculated (before any seam removed)
	global firstCalculation
	if firstCalculation == True:
		PREWITT_EDGE_PATH = EDGE_DETECTION_PATH + "prewitt.jpg"
		cv2.imwrite(PREWITT_EDGE_PATH, np.rot90(prewitt, 3, (0, 1)))

		PREWITT_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_prewitt.jpg"
		cv2.imwrite(PREWITT_ENERGY_PATH, np.rot90(energy_map, 3, (0, 1)))
		firstCalculation = False

	return energy_map

def laplacian(img):
	kernelL = np.array([
	    [0.0, 1.0, 0.0],
	    [1.0, -4.0, 1.0],
	    [0.0, 1.0, 0.0],
	])
	kernelL = np.stack([kernelL] * 3, axis=2)
	img = img.astype('float32')
	laplacian = np.absolute(convolve(img, kernelL))

	energy_map = laplacian.sum(axis=2)

	global firstCalculation
	if firstCalculation == True:
		LAPLACIAN_EDGE_PATH = EDGE_DETECTION_PATH + "laplacian.jpg"
		cv2.imwrite(LAPLACIAN_EDGE_PATH, np.rot90(laplacian, 3, (0, 1)))

		LAPLACIAN_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_laplacian.jpg"
		cv2.imwrite(LAPLACIAN_ENERGY_PATH, np.rot90(energy_map, 3, (0, 1)))
		firstCalculation = False
	return energy_map

def roberts(img):
	kernelR1 = np.array([
	    [1.0, 0.0],
	    [0.0, -1.0],
	])
	kernelR1 = np.stack([kernelR1] * 3, axis=2)

	kernelR2 = np.array([
	    [0.0, 1.0],
	    [-1.0, 0.0],
	])
	kernelR2 = np.stack([kernelR2] * 3, axis=2)

	img = img.astype('float32')
	roberts = np.absolute(convolve(img, kernelR1)) + np.absolute(convolve(img, kernelR2))

	energy_map = roberts.sum(axis=2)

	#Saves the first energy map calculated (before any seam removed)
	global firstCalculation
	if firstCalculation == True:
		ROBERTS_EDGE_PATH = EDGE_DETECTION_PATH + "roberts.jpg"
		cv2.imwrite(ROBERTS_EDGE_PATH, np.rot90(roberts, 3, (0, 1)))

		ROBERTS_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_roberts.jpg"
		cv2.imwrite(ROBERTS_ENERGY_PATH, np.rot90(energy_map, 3, (0, 1)))
		firstCalculation = False

	return energy_map


#Main program
if __name__ == '__main__':
	ap = argparse.ArgumentParser()

	ap.add_argument("-in", help="Path to input image", required=True)
	ap.add_argument("-out", help="Output image file name", required=True)
	ap.add_argument("-scale", help="Downsizing scale. e.g. 0.5", required=True, type=float, default=0.5)
	ap.add_argument("-seam", help="Seam orientation (h = horizontal seam, v = vertical seam", required=True)
	ap.add_argument("-energy", help="Energy algorithm (s = Sobel, p = Prewitt)", required=False, default='s')
	args = vars(ap.parse_args())

	IMG_NAME, OUTPUT_NAME, SCALE, SEAM_ORIENTATION, ENERGY_ALGORITHM = args["in"], args["out"], args["scale"], args["seam"], args["energy"]

	#create results directory
	path("../results").mkdir(parents=True, exist_ok=True)
	path("../results/resized_images/").mkdir(parents=True, exist_ok=True)
	path("../results/edge_detection_images/").mkdir(parents=True, exist_ok=True)
	path("../results/energy_maps/").mkdir(parents=True, exist_ok=True)

	#paths definition
	IMG_PATH = "../images/" + IMG_NAME
	OUTPUT_PATH = "../results/resized_images/" + OUTPUT_NAME
	ENERGY_MAP_PATH = "../results/energy_maps/"
	EDGE_DETECTION_PATH = "../results/edge_detection_images/"
	img = cv2.imread(IMG_PATH)

	#Sets the boolean used to save the energy map
	firstCalculation = True

	if SEAM_ORIENTATION == 'h':
		img = np.rot90(img, 1, (0, 1))
		img = resize(img, SCALE)
		out = np.rot90(img, 3, (0, 1))
	elif SEAM_ORIENTATION == 'v':
		out = resize(img, SCALE)
	else:
		print("Error: invalid arguments. Use -h argument for help")
		sys.exit(1)


	OUTPUT_PATH = "../results/resized_images/" + OUTPUT_NAME
	cv2.imwrite(OUTPUT_PATH, out)
