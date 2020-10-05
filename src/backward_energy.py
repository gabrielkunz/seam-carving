# pylint: disable=E1101

import cv2
import numpy as np
from scipy import ndimage
from scipy import misc
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
CANNY_EDGE_PATH = EDGE_DETECTION_PATH + "canny.jpg"
CANNY_ENERGY_PATH = ENERGY_MAP_PATH + "em_canny.jpg"


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

        self.canny_edge_detector = self.CannyEdgeDetector(input_image)

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
        sobel = np.absolute(convolve(img, kernel_sx)) + \
            np.absolute(convolve(img, kernel_sy))

        # We sum the energies in the red, green, and blue channels
        self.energy_map = sobel.sum(axis=2)

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(SOBEL_EDGE_PATH, np.rot90(sobel, 3, (0, 1)))
            cv2.imwrite(SOBEL_ENERGY_PATH, np.rot90(
                self.energy_map, 3, (0, 1)))

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
        prewitt = np.absolute(convolve(img, kernel_px)) + \
            np.absolute(convolve(img, kernel_py))

        self.energy_map = prewitt.sum(axis=2)

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(PREWITT_EDGE_PATH, np.rot90(prewitt, 3, (0, 1)))
            cv2.imwrite(PREWITT_ENERGY_PATH, np.rot90(
                self.energy_map, 3, (0, 1)))

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
            cv2.imwrite(LAPLACIAN_ENERGY_PATH, np.rot90(
                self.energy_map, 3, (0, 1)))

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
        roberts = np.absolute(convolve(img, kernel_r1)) + \
            np.absolute(convolve(img, kernel_r2))

        self.energy_map = roberts.sum(axis=2)

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(ROBERTS_EDGE_PATH, np.rot90(roberts, 3, (0, 1)))
            cv2.imwrite(ROBERTS_ENERGY_PATH, np.rot90(
                self.energy_map, 3, (0, 1)))

        return self.energy_map

    def canny(self, img):
        self.energy_map = self.canny_edge_detector.detect(img=img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)

        return self.energy_map

    class CannyEdgeDetector:
        """
        Canny edge detection.
        Code adapted from
        https://github.com/FienSoP/canny_edge_detector
        """

        def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
            self.img = img
            self.energy_map = []
            self.img_smoothed = None
            self.gradient_mat = None
            self.theta_mat = None
            self.non_max_img = None
            self.threshold_img = None
            self.weak_pixel = weak_pixel
            self.strong_pixel = strong_pixel
            self.sigma = sigma
            self.kernel_size = kernel_size
            self.low_threshold = lowthreshold
            self.high_threshold = highthreshold
            return

        def gaussian_kernel(self, size, sigma=1):
            size = int(size) // 2
            x, y = np.mgrid[-size:size+1, -size:size+1]
            normal = 1 / (2.0 * np.pi * sigma**2)
            g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
            return g

        def sobel_filters(self, img):
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
            Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

            Ix = ndimage.filters.convolve(img, Kx)
            Iy = ndimage.filters.convolve(img, Ky)

            G = np.hypot(Ix, Iy)
            G = G / G.max() * 255
            theta = np.arctan2(Iy, Ix)
            return (G, theta)

        def non_max_suppression(self, img, D):
            M, N = img.shape
            Z = np.zeros((M, N), dtype=np.int32)
            angle = D * 180. / np.pi
            angle[angle < 0] += 180

            for i in range(1, M-1):
                for j in range(1, N-1):
                    try:
                        q = 255
                        r = 255

                        # angle 0
                        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                            q = img[i, j+1]
                            r = img[i, j-1]
                        # angle 45
                        elif (22.5 <= angle[i, j] < 67.5):
                            q = img[i+1, j-1]
                            r = img[i-1, j+1]
                        # angle 90
                        elif (67.5 <= angle[i, j] < 112.5):
                            q = img[i+1, j]
                            r = img[i-1, j]
                        # angle 135
                        elif (112.5 <= angle[i, j] < 157.5):
                            q = img[i-1, j-1]
                            r = img[i+1, j+1]

                        if (img[i, j] >= q) and (img[i, j] >= r):
                            Z[i, j] = img[i, j]
                        else:
                            Z[i, j] = 0

                    except IndexError as _:
                        pass

            return Z

        def threshold(self, img):

            high_threshold = img.max() * self.high_threshold
            low_threshold = high_threshold * self.low_threshold

            M, N = img.shape
            res = np.zeros((M, N), dtype=np.int32)

            weak = np.int32(self.weak_pixel)
            strong = np.int32(self.strong_pixel)

            strong_i, strong_j = np.where(img >= high_threshold)
            zeros_i, zeros_j = np.where(img < low_threshold)

            weak_i, weak_j = np.where(
                (img <= high_threshold) & (img >= low_threshold))

            res[strong_i, strong_j] = strong
            res[weak_i, weak_j] = weak

            return (res)

        def hysteresis(self, img):

            M, N = img.shape
            weak = self.weak_pixel
            strong = self.strong_pixel

            for i in range(1, M-1):
                for j in range(1, N-1):
                    if (img[i, j] == weak):
                        try:
                            if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                                or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                                img[i, j] = strong
                            else:
                                img[i, j] = 0
                        except IndexError as _:
                            pass

            return img

        def detect(self, img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100):
            img_final = []

            for _, img in enumerate(self.img):
                self.img_smoothed = convolve(
                    img, self.gaussian_kernel(self.kernel_size, self.sigma))
                self.gradient_mat, self.theta_mat = self.sobel_filters(
                    self.img_smoothed)
                self.non_max_img = self.non_max_suppression(
                    self.gradient_mat, self.theta_mat)
                self.threshold_img = self.threshold(self.non_max_img)
                img_final = self.hysteresis(self.threshold_img)
                
            img_final = np.stack([img_final] * 3, axis=2)
            self.energy_map = img_final.sum(axis=2)

            cv2.imwrite(CANNY_EDGE_PATH, np.rot90(img_final, 3, (0, 1)))
            cv2.imwrite(CANNY_ENERGY_PATH, np.rot90(
                self.energy_map, 3, (0, 1)))

            return self.energy_map
