import numpy as np
import cv2


def gaussuian_filter(height, width, sigma=1, muu=0):
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size

    x, y = np.meshgrid(np.linspace(-1, 1, width),
                       np.linspace(-1, 1, height))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    normal = 1 / (2.0 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    g = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal
    g = (g / np.max(g))[..., np.newaxis]
    return np.concatenate([g, g, g], axis=-1)


def rescale(im: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = im.shape
    if scale == 1.0:
        return im
    return cv2.resize(im, (int(w * scale), int(h * scale)))
