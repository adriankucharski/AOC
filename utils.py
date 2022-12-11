
import cv2
import numpy as np


def rescale(im: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = im.shape
    if scale == 1.0:
        return im
    return cv2.resize(im, (int(w * scale), int(h * scale)))