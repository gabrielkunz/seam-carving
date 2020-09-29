# pylint: disable=E1101

import cv2
import numpy as np
from scipy.ndimage.filters import convolve

# Paths definition
ENERGY_MAP_PATH = "../results/energy_maps/"
EDGE_DETECTION_PATH = "../results/edge_detection_images/"
SOBEL_EDGE_PATH = EDGE_DETECTION_PATH + "sobel.jpg"
SOBEL_ENERGY_PATH = ENERGY_MAP_PATH + "em_sobel.jpg"
PREWITT_EDGE_PATH = EDGE_DETECTION_PATH + "prewitt.jpg"
PREWITT_ENERGY_PATH = ENERGY_MAP_PATH + "em_prewitt.jpg"
LAPLACIAN_EDGE_PATH = EDGE_DETECTION_PATH + "laplacian.jpg"
LAPLACIAN_ENERGY_PATH = ENERGY_MAP_PATH + "em_laplacian.jpg"
ROBERTS_EDGE_PATH = EDGE_DETECTION_PATH + "roberts.jpg"
ROBERTS_ENERGY_PATH = ENERGY_MAP_PATH + "em_roberts.jpg"

class BackwardEnergy:
    """
    Backaward Energy mapping with different edge detection algorithms
    """
    def __init__(self, input_image, seam_orientation):
        self.energy_map = []
        if seam_orientation == 'h':
            self.input_image = np.rot90(input_image, 1, (0, 1))
        else:
            self.input_image = input_image

    def sobel(self, img):
        """
        Sobel edge detection
        """
        kernel_sy = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        kernel_sy = np.stack([kernel_sy] * 3, axis=2)

        kernel_sx = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        kernel_sx = np.stack([kernel_sx] * 3, axis=2)

        img = img.astype('float32')
        sobel = np.absolute(convolve(img, kernel_sx)) + np.absolute(convolve(img, kernel_sy))

        # We sum the energies in the red, green, and blue channels
        self.energy_map = sobel.sum(axis=2)

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(SOBEL_EDGE_PATH, np.rot90(sobel, 3, (0, 1)))
            cv2.imwrite(SOBEL_ENERGY_PATH, np.rot90(self.energy_map, 3, (0, 1)))

        return self.energy_map

    def prewitt(self, img):
        """
        Prewitt edge detection
        """
        kernel_px = np.array([
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ])
        kernel_px = np.stack([kernel_px] * 3, axis=2)

        kernel_py = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0],
        ])
        kernel_py = np.stack([kernel_py] * 3, axis=2)

        img = img.astype('float32')
        prewitt = np.absolute(convolve(img, kernel_px)) + np.absolute(convolve(img, kernel_py))

        self.energy_map = prewitt.sum(axis=2)

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(PREWITT_EDGE_PATH, np.rot90(prewitt, 3, (0, 1)))
            cv2.imwrite(PREWITT_ENERGY_PATH, np.rot90(self.energy_map, 3, (0, 1)))

        return self.energy_map

    def laplacian(self, img):
        """
        Laplacian edge detection
        """
        kernel_l = np.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        kernel_l = np.stack([kernel_l] * 3, axis=2)
        img = img.astype('float32')
        laplacian = np.absolute(convolve(img, kernel_l))

        self.energy_map = laplacian.sum(axis=2)

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(LAPLACIAN_EDGE_PATH, np.rot90(laplacian, 3, (0, 1)))
            cv2.imwrite(LAPLACIAN_ENERGY_PATH, np.rot90(self.energy_map, 3, (0, 1)))

        return self.energy_map

    def roberts(self, img):
        """
        Roberts edge detection
        """
        kernel_r1 = np.array([
            [1.0, 0.0],
            [0.0, -1.0],
        ])
        kernel_r1 = np.stack([kernel_r1] * 3, axis=2)

        kernel_r2 = np.array([
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        kernel_r2 = np.stack([kernel_r2] * 3, axis=2)

        img = img.astype('float32')
        roberts = np.absolute(convolve(img, kernel_r1)) + np.absolute(convolve(img, kernel_r2))

        self.energy_map = roberts.sum(axis=2)

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(ROBERTS_EDGE_PATH, np.rot90(roberts, 3, (0, 1)))
            cv2.imwrite(ROBERTS_ENERGY_PATH, np.rot90(self.energy_map, 3, (0, 1)))

        return self.energy_map
