import cv2
import numpy as np

KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def generate_boundary_xray(mask):
    mask = cv2.GaussianBlur(mask * 255, (5, 5), 3)
    mask = mask / 255
    xray = (4 * np.multiply(mask, 1 - mask))
    return xray


def generate_boundary_cv2(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, KERNEL_5)
    return mask