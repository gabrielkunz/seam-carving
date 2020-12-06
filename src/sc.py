# pylint: disable=E1101

import sys
import os
import argparse
import warnings
from pathlib import Path as path
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
from tqdm import trange
from numba import jit

from backward_energy import BackwardEnergy
from forward_energy import ForwardEnergy

rc = {"figure.constrained_layout.use": True,
      "axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": True,
      "axes.spines.top": False,
      "xtick.bottom": False,
      "xtick.labelbottom": True,
      "ytick.labelleft": True,
      "ytick.left": False}
plt.rcParams.update(rc)

# This is to ignore NumbaWarnings and NumbaDeprecationWarnings issued by @jit
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Seam carving calling function
def resize(img, scale):
    """
    Defines the new image shape based on scale provided
    """
    _, columns, _ = img.shape
    new_columns = int(scale * columns)

    for i in trange(columns - new_columns):
        img = seamCarving(img)

    return img

# Seam carving functions
@jit
def seamCarving(img):
    """
    Removes the seam selected (carving process)
    """
    rows, columns, _ = img.shape

    M, backtrack = findSeam(img)

    # creates a mask with value True in all positions
    mask = np.ones((rows, columns), dtype=np.bool)

    # finds the position of the smalletst element in the last row of M
    j = np.argmin(M[-1])
    # from bottom-up
    for i in reversed(range(rows)):
        # marks the pixel for deletion
        mask[i, j] = False
        # gets the column position from the backtrack matrix
        j = backtrack[i, j]

    # converts the mask to 3D since the image has 3 channels
    mask = np.stack([mask] * 3, axis=2)

    # deletes the flagged pixels and resize the image to the new dimension
    img = img[mask].reshape((rows, columns - 1, 3))
    return img

@jit
def findSeam(img):
    """
    Finds the minimal energy path (seam) to be removed from the image
    """

    rows, columns, _ = img.shape  # m = rows, n = columns

    # calculates the energy of each pixel using edge detection algorithms. e.g. Sobel, Prewitt, etc.
    energy_map = calculateEnergy(img)

    # the energy map is copied into M
    M = energy_map.copy()
    # creates the backtrack to find the list of pixels present in the found seam
    # backtrack is a matrix of zeroes with the same dimensions as the image/energy map/M
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, rows):
        for j in range(0, columns):
            # if we are in the first column (the one more to the left)
            if j == 0:
                # index contains the minimal between M(i-1,j) and M(i-1,j+2) (pq nao j+1???)
                # trocar por -1 e ver se tem alguma diferenca
                index = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = index + j
                minimal_energy = M[i - 1, index + j]
            # if we are in the other columns
            else:
                # index contains the minimal between M(i-1,j-1), M(i-1,j) and M(i-1,j+2)
                # trocar por -1 também e ver
                index = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = index + j - 1
                minimal_energy = M[i-1, index+j-1]

            M[i, j] += minimal_energy

    return M, backtrack

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
        energy_map = fe.fast_forward_energy(img)
    elif ENERGY_ALGORITHM == 'c':
        energy_map = be.canny(img)
    else:
        energy_map = be.sobel(img)

    return energy_map

# Other functions
def standardResize(img, scale, seam_orientation):
    """
    Resize the image without considering its content
    """
    if seam_orientation == 'h':
        width = img.shape[1]
        height = int(img.shape[0] * scale)
        dsize = (width, height)
        std_resize_image = cv2.resize(img, dsize)
    elif seam_orientation == 'v':
        width = int(img.shape[1]*scale)
        height = img.shape[0]
        dsize = (width, height)
        std_resize_image = cv2.resize(img, dsize)

    return std_resize_image

def plotResult(img, out, std_resize_image, energyFunction, scale, img_name):
    """
    Plot the original image with its mean energy, energy map and
    the resized image with its mean energy.
    """

    fig = plt.figure(figsize=(10, 10))
    # Fix message blinking when hover
    fig.canvas.toolbar.set_message = lambda x: ""

    plt.subplot(221)
    plt.imshow(img)
    plt.title('Original image')

    plt.subplot(222)
    plt.imshow(energyFunction(img))
    plt.title('Energy map - ' + (energyFunction.__name__).capitalize())

    plt.subplot(223)
    plt.imshow(std_resize_image)
    plt.title('Standard Resize result (resize scale = ' + str(scale) + ')')

    plt.subplot(224)
    plt.imshow(out)
    plt.title('Seam Carving result (resize scale = ' + str(scale) + ')')

    plt.tight_layout()

    figure = plt.gcf()
    plt.show()

    fig_name = "plt_" + energyFunction.__name__ + "_" + img_name[:-4]
    fig_path = "../results/plot_figures/" + fig_name
    figure.savefig(fig_path + ".png")

def metrics(img, out, std_resize_image, img_name, energyFunction, metrics_file):
    """
    Calculate the metrics for comparison between each energy mapping
    method used for the seam carving. The metrics used are:
    - Mean energy
    - Shannon entropy
    - Mutual information
    """

    img_mean_energy = np.mean(img)
    std_mean_energy = np.mean(std_resize_image)
    out_mean_energy = np.mean(out)

    img_entropy = skimage.measure.shannon_entropy(img)
    std_entropy = skimage.measure.shannon_entropy(std_resize_image)
    out_entropy = skimage.measure.shannon_entropy(out)

    with open('../results/metrics.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([img_name, (energyFunction.__name__).capitalize(), img_mean_energy, std_mean_energy, out_mean_energy, img_entropy, std_entropy, out_entropy])

    with open('../results/metrics_excel.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([img_name, (energyFunction.__name__).capitalize(), img_mean_energy, std_mean_energy, out_mean_energy, img_entropy, std_entropy, out_entropy])

# Main program
if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-in", help="Path to input image", required=True)
    ap.add_argument("-scale", help="Downsizing scale. e.g. 0.5", required=True, type=float,
                    default=0.5)
    ap.add_argument("-seam", help="Seam orientation (h = horizontal seam, v = vertical seam",
                    required=True)
    ap.add_argument(
        "-energy", help="Energy mapping algorithm (s = Sobel, p = Prewitt, l = Laplacian, r = Roberts, f = Forward energy)", required=False, default='s')
    ap.add_argument("-plot", help="Plot result after resizing",
                    action='store_true')
    ap.add_argument("-metrics", help="Save metrics in a .csv file", action='store_true')
    args = vars(ap.parse_args())

    IMG_NAME, SCALE, SEAM_ORIENTATION, ENERGY_ALGORITHM = args[
        "in"], args["scale"], args["seam"], args["energy"]

    # Create results directory
    path("../results").mkdir(parents=True, exist_ok=True)
    path("../results/resized_images/").mkdir(parents=True, exist_ok=True)
    path("../results/edge_detection_images/").mkdir(parents=True, exist_ok=True)
    path("../results/energy_maps/").mkdir(parents=True, exist_ok=True)
    path("../results/plot_figures/").mkdir(parents=True, exist_ok=True)

    # Create .csv file for metrics if requested by the user
    if args["metrics"]:
        metrics_file = open("../results/metrics.csv",'w')
        metrics_excel_file = open("../results/metrics_excel.csv",'w')

        with open('../results/metrics.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Image filename", "Energy mapping algorithm", "Original mean energy", "Standard resize mean energy", "Seam carving mean energy", "Original entropy", "Strandard resize entropy", "Seam carving entropy"])

        with open('../results/metrics_excel.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Image filename", "Energy mapping algorithm", "Original mean energy", "Standard resize mean energy", "Seam carving mean energy", "Original entropy", "Strandard resize entropy", "Seam carving entropy"])

    # paths definition
    IMG_PATH = "../images/" + IMG_NAME

    input_image = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
    std_resize_image = standardResize(input_image, SCALE, SEAM_ORIENTATION)

    # Instantiate the energy mapping classes
    be = BackwardEnergy(input_image, SEAM_ORIENTATION)
    fe = ForwardEnergy(input_image, SEAM_ORIENTATION)

    # Number of diferent resize algorithms existing in this program
    ALGORITHMS = ['s', 'p', 'l', 'r', 'c', 'f']
    ENERGY_MAPPING_FUNCTIONS = [be.sobel,
                                be.prewitt,
                                be.laplacian,
                                be.roberts,
                                be.canny,
                                fe.fast_forward_energy]

    # Run program for all the energy mapping algorithms implemented
    if ENERGY_ALGORITHM == "all":
        for a in ALGORITHMS:
            ENERGY_ALGORITHM = a
            energyFunction = ENERGY_MAPPING_FUNCTIONS[ALGORITHMS.index(a)]

            if SEAM_ORIENTATION == 'h':
                print("Performing seam carving with energy mapping function "
                      + energyFunction.__name__ + "()...")

                img = np.rot90(input_image, 1, (0, 1))
                img = resize(img, SCALE)
                out = np.rot90(img, 3, (0, 1))
            elif SEAM_ORIENTATION == 'v':
                print("Performing seam carving with energy mapping function "
                      + energyFunction.__name__ + "()...")

                out = resize(input_image, SCALE)
            else:
                print("Error: invalid arguments. Use -h argument for help")
                sys.exit(1)

            # Plot the result if requested by the user
            if args["plot"]:
                plotResult(input_image, out, std_resize_image , energyFunction, SCALE, IMG_NAME)

            # Calculate and save metrics if requested by the user
            if args["metrics"]:
                metrics(input_image, out, std_resize_image, IMG_NAME, energyFunction, metrics_file)

            print("Seam carving with energy energy mapping function "
                  + energyFunction.__name__ + "() completed.")

            OUTPUT_PATH = "../results/resized_images/" + \
                (os.path.splitext(IMG_NAME)[0]) + \
                "_" + ENERGY_ALGORITHM + ".jpg"
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(OUTPUT_PATH, out)

    else:  # Run program for a specific algorithm

        energyFunction = ENERGY_MAPPING_FUNCTIONS[ALGORITHMS.index(
            ENERGY_ALGORITHM)]

        if SEAM_ORIENTATION == 'h':
            img = np.rot90(input_image, 1, (0, 1))
            img = resize(img, SCALE)
            out = np.rot90(img, 3, (0, 1))
        elif SEAM_ORIENTATION == 'v':
            out = resize(input_image, SCALE)
        else:
            print("Error: invalid arguments. Use -h argument for help")
            sys.exit(1)

        # Plot the result if requested by the user
        if args["plot"]:
            plotResult(input_image, out, std_resize_image, energyFunction, SCALE, IMG_NAME)

        # Calculate and save metrics if requested by the user
        if args["metrics"]:
            metrics(input_image, out, std_resize_image , energyFunction, metrics_file)
        
        print("Seam carving with energy mapping function "
              + energyFunction.__name__ + " completed.")

        OUTPUT_PATH = "../results/resized_images/" + \
            (os.path.splitext(IMG_NAME)[0]) + "_" + ENERGY_ALGORITHM + ".jpg"
        
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_PATH, out)
