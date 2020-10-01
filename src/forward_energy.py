# pylint: disable=E1101

import cv2
import numpy as np

# Path definition
ENERGY_MAP_PATH = "../results/energy_maps/"
FORWARD_ENERGY_PATH = ENERGY_MAP_PATH + "em_forwardEnergy.jpg"


class ForwardEnergy:
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.

    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving
    """

    def __init__(self, input_image, seam_orientation):
        self.input_image = input_image
        self.energy_map = []
        if seam_orientation == 'h':
            self.input_image = np.rot90(input_image, 1, (0, 1))
        else:
            self.input_image = input_image

    def fast_forward_energy(self, img):
        h, w = img.shape[:2]
        img = cv2.cvtColor(img.astype(np.uint8),
                           cv2.COLOR_BGR2GRAY).astype(np.float64)

        energy_map = np.zeros((h, w))
        m = np.zeros((h, w))

        U = np.roll(img, 1, axis=0)
        L = np.roll(img, 1, axis=1)
        R = np.roll(img, -1, axis=1)

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

            argmins = np.argmin(mULR, axis=0)
            m[i] = np.choose(argmins, mULR)
            energy_map[i] = np.choose(argmins, cULR)

        self.energy_map = energy_map

        # If the image being handled has the same number of columns as
        # the input image, then this is the first iteration
        if img.shape[1] == self.input_image.shape[1]:
            cv2.imwrite(FORWARD_ENERGY_PATH, np.rot90(
                self.energy_map, 3, (0, 1)))

        return self.energy_map
