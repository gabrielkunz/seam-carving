# pylint: disable=E1101

import numpy as np
from scipy.ndimage.filters import convolve

class BackwardEnergy:
    """
    Backaward Energy mapping with different edge detection algorithms
    """
    def __init__(self):
        self.energy_map = []

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

        return self.energy_map
