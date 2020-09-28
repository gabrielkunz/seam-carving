import numpy as np
from scipy.ndimage.filters import convolve

class BackwardEnergy(object):
    """
    Backaward Energy mapping with different edge detection algorithms
    """

    def __init__(self, arg):
        super(, self).__init__()
        self.arg = arg

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

        return energy_map
