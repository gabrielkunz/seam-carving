# pylint: disable=E1101

import sys
import argparse
import warnings
from pathlib import Path as path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from numba import jit
from scipy.ndimage.filters import convolve

rc = {"figure.constrained_layout.use" : True,
      "axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)

#This is to ignore NumbaWarnings and NumbaDeprecationWarnings issued by @jit
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def resize(img, scale):
    """
    Defines the new image shape based on scale provided
    """
    m,n, _ = img.shape # m = rows, n = columns
    new_n = int(scale * n)

    for i in trange(n - new_n):
        img = seamCarving(img)

    return img

#Seam carving functions
@jit
def seamCarving(img):
    """
    Removes the seam selected (carving process)
    """
    m, n, _ = img.shape # m = rows, n = columns

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
def findSeam(img):
    """
    Finds the minimal energy path (seam) to be removed from the image
    """

    m,n, _ = img.shape # m = rows, n = columns

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
                #index contains the minimal between M(i-1,j-1), M(i-1,j) and M(i-1,j+2)
                index = np.argmin(M[i - 1, j - 1:j + 2]) #trocar por -1 também e ver
                backtrack[i,j] = index + j - 1
                minimal_energy = M[i-1, index+j-1]

            M[i,j] += minimal_energy

    return M, backtrack

@jit
def calculateEnergy(img):
    """
    Calculates the energy map using edge detection algorithms for backward energy
    or the forward energy algorithm
    """
    if ENERGY_ALGORITHM == 's':
        energy_map = sobel(img)
    elif ENERGY_ALGORITHM == 'p':
        energy_map = prewitt(img)
    elif ENERGY_ALGORITHM == 'l':
        energy_map = laplacian(img)
    elif ENERGY_ALGORITHM == 'r':
        energy_map = roberts(img)
    elif ENERGY_ALGORITHM == 'f':
        energy_map = forwardEnergy(img)
    else:
        energy_map = sobel(img)

    return energy_map

#Energy mapping functions
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
        cv2.imwrite(SOBEL_EDGE_PATH, np.rot90(sobel, 3, (0, 1)))
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
        cv2.imwrite(PREWITT_EDGE_PATH, np.rot90(prewitt, 3, (0, 1)))
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
        cv2.imwrite(LAPLACIAN_EDGE_PATH, np.rot90(laplacian, 3, (0, 1)))
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
        cv2.imwrite(ROBERTS_EDGE_PATH, np.rot90(roberts, 3, (0, 1)))
        cv2.imwrite(ROBERTS_ENERGY_PATH, np.rot90(energy_map, 3, (0, 1)))
        firstCalculation = False

    return energy_map

def forwardEnergy(img):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.

    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving
    """
    h, w = img.shape[:2]
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy_map = np.zeros((h,w))
    m = np.zeros((h,w))

    U = np.roll(img, 1, axis = 0)
    L = np.roll(img, 1, axis = 1)
    R = np.roll(img, -1, axis = 1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis = 0)
        m[i] = np.choose(argmins, mULR)
        energy_map[i] = np.choose(argmins, cULR)

    #Saves the first energy map calculated (before any seam removed)
    global firstCalculation
    if firstCalculation == True:
        cv2.imwrite(FORWARD_ENERGY_PATH, np.rot90(energy_map, 3, (0, 1)))
        firstCalculation = False

    return energy_map

#Other functions
def plotResult(img, out, energyFunction):
    """
    Plot the original image with its mean energy, energy map and
    the resized image with its mean energy.
    """

    fig = plt.figure(figsize=(10, 10))
    ##Fix message blinking when hover
    fig.canvas.toolbar.set_message = lambda x: ""

    plt.subplot(211)
    plt.imshow(img)
    plt.title('Original Image\n'+'Mean energy = ' + str(np.mean(img)))

    plt.subplot(223)
    plt.imshow(energyFunction(img))
    plt.title('Energy Map (' + energyFunction.__name__ + ')')

    plt.subplot(224)
    plt.imshow(out)
    plt.title('Carving Result\n'+'Mean energy = ' + str(np.mean(out)))

    plt.show()

#Main program
if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-in", help="Path to input image", required=True)
    ap.add_argument("-scale", help="Downsizing scale. e.g. 0.5", required=True, type=float,
                    default=0.5)
    ap.add_argument("-seam", help="Seam orientation (h = horizontal seam, v = vertical seam",
                    required=True)
    ap.add_argument("-energy", help="Energy mapping algorithm (s = Sobel, p = Prewitt, l = Laplacian, r = Roberts, f = Forward energy)", required=False, default='s')
    ap.add_argument("-plot", help="Plot result after resizing", action='store_true')
    args = vars(ap.parse_args())

    IMG_NAME, SCALE, SEAM_ORIENTATION, ENERGY_ALGORITHM = args["in"], args["scale"], args["seam"], args["energy"]

    #create results directory
    path("../results").mkdir(parents=True, exist_ok=True)
    path("../results/resized_images/").mkdir(parents=True, exist_ok=True)
    path("../results/edge_detection_images/").mkdir(parents=True, exist_ok=True)
    path("../results/energy_maps/").mkdir(parents=True, exist_ok=True)

    #Number of diferent resize algorithms existing in this program
    ALGORITHMS = ['s', 'p', 'l', 'r', 'f']
    ENERGY_MAPPING_FUNCTIONS = [sobel,
                                prewitt,
                                laplacian,
                                roberts,
                                forwardEnergy]

    #paths definition
    IMG_PATH = "../images/" + IMG_NAME
    ENERGY_MAP_PATH = "../results/energy_maps/"
    EDGE_DETECTION_PATH = "../results/edge_detection_images/"
    SOBEL_EDGE_PATH = EDGE_DETECTION_PATH + "sobel.jpg"
    SOBEL_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_sobel.jpg"
    PREWITT_EDGE_PATH = EDGE_DETECTION_PATH + "prewitt.jpg"
    PREWITT_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_prewitt.jpg"
    LAPLACIAN_EDGE_PATH = EDGE_DETECTION_PATH + "laplacian.jpg"
    LAPLACIAN_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_laplacian.jpg"
    ROBERTS_EDGE_PATH = EDGE_DETECTION_PATH + "roberts.jpg"
    ROBERTS_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_roberts.jpg"
    FORWARD_ENERGY_PATH = ENERGY_MAP_PATH + "energy_map_forwardEnergy.jpg"

    imgOriginal = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)

    #Sets the boolean used to save the energy map
    firstCalculation = True

    #Run program for all the energy mapping algorithms implemented
    if ENERGY_ALGORITHM == "all":
        for a in ALGORITHMS:
            ENERGY_ALGORITHM = a
            energyFunction = ENERGY_MAPPING_FUNCTIONS[ALGORITHMS.index(a)]

            if SEAM_ORIENTATION == 'h':
                print("Performing seam carving with energy mapping function "
                        + energyFunction.__name__ + "()...")
                img = np.rot90(imgOriginal, 1, (0, 1))
                img = resize(img, SCALE)
                out = np.rot90(img, 3, (0, 1))
            elif SEAM_ORIENTATION == 'v':
                print("Performing seam carving with energy mapping function "
                        + energyFunction.__name__ + "()...")
                out = resize(imgOriginal, SCALE)
            else:
                print("Error: invalid arguments. Use -h argument for help")
                sys.exit(1)

            #Plot the result if requested by the user
            if args["plot"]:
                plotResult(imgOriginal, out, energyFunction)

            print("Seam carving with energy energy mapping function "
                    + energyFunction.__name__ + "() completed.")

            OUTPUT_PATH = "../results/resized_images/" + ENERGY_ALGORITHM + ".jpg"
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(OUTPUT_PATH, out)

            #Resets the boolean to save the energy map for the other algorithms
            firstCalculation = True

    else: #Run program for a specific algorithm

        energyFunction = ENERGY_MAPPING_FUNCTIONS[ALGORITHMS.index(ENERGY_ALGORITHM)]

        if SEAM_ORIENTATION == 'h':
            img = np.rot90(imgOriginal, 1, (0, 1))
            img = resize(img, SCALE)
            out = np.rot90(img, 3, (0, 1))
        elif SEAM_ORIENTATION == 'v':
            out = resize(imgOriginal, SCALE)
        else:
            print("Error: invalid arguments. Use -h argument for help")
            sys.exit(1)
        #Plot the result if requested by the user
        if args["plot"]:
            plotResult(imgOriginal, out, energyFunction)

        print("Seam carving with energy mapping function "
                + energyFunction.__name__ + " completed.")

        OUTPUT_PATH = "../results/resized_images/" + ENERGY_ALGORITHM + ".jpg"
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_PATH, out)
