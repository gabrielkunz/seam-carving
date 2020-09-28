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

import backward_energy as be
import forward_energy as fe

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
        energy_map = be.sobel(img)
    elif ENERGY_ALGORITHM == 'p':
        energy_map = be.prewitt(img)
    elif ENERGY_ALGORITHM == 'l':
        energy_map = be.laplacian(img)
    elif ENERGY_ALGORITHM == 'r':
        energy_map = be.roberts(img)
    elif ENERGY_ALGORITHM == 'f':
        energy_map = fe.forwardEnergy(img)
    else:
        energy_map = be.sobel(img)

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
    ENERGY_MAPPING_FUNCTIONS = [be.sobel,
                                be.prewitt,
                                be.laplacian,
                                be.roberts,
                                fe.forwardEnergy]

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
