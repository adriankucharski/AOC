from multiprocessing.pool import ThreadPool
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple
import cv2
import itertools
import numba
import time
from sklearn.metrics import mutual_info_score
from skimage import metrics

# 4000x2000 -> 400x200
# 400x200 -> [x, y, w, h]
# [x, y, w, h] -> [x1, y1, w1, h1]
# 4000x2000

def gaussuian_filter(height, width, sigma=1, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, width),
                       np.linspace(-1, 1, height))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    g = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    g = (g / np.max(g))[..., np.newaxis]
    return np.concatenate([g, g, g], axis=-1)
 
gauss_kernel = None



def func(subObject: np.ndarray, selectedObject: np.ndarray) -> float:
    global gauss_kernel
    if gauss_kernel is None:
        h, w, _ = subObject.shape
        gauss_kernel = gaussuian_filter(h, w, 1)
    # return 1/ metrics.peak_signal_noise_ratio(subObject, selectedObject) 
    # return metrics.normalized_mutual_information(subObject, selectedObject)
    # subObject = cv2.cvtColor(subObject, cv2.COLOR_BGR2GRAY)
    # selectedObject = cv2.cvtColor(selectedObject, cv2.COLOR_BGR2GRAY)
    
    # subObject = subObject / gauss_kernel
    # selectedObject = selectedObject / gauss_kernel
    return 1/metrics.normalized_mutual_information(subObject, selectedObject)
    # return np.sum((subObject - selectedObject) ** 2)


class ObjectTracker:
    def __init__(
        self,
        box: Tuple[int, int, int, int],
        frame: np.ndarray,
        stride: int = 4,
        margin: int = 30,
        
    ) -> None:
        self.x, self.y, self.h, self.w = box
        self.selectedObject = frame[self.x : self.x + self.w, self.y : self.y + self.h]
        self.stride = stride
        self.margin = margin
        self.pool = ThreadPool(multiprocessing.cpu_count())
        self.first = np.array(self.selectedObject)

    def trackObject(self, nextFrame: np.ndarray) -> np.ndarray:
        vw, vh, _ = nextFrame.shape
        subObjects: List[np.ndarray] = []
        indexes: List[Tuple[int, int]] = []
        for i in range(
            max(0, self.x - self.margin),
            min(vw - self.w, self.x + self.margin),
            self.stride,
        ):
            for j in range(
                max(0, self.y - self.margin),
                min(vh - self.h, self.y + self.margin),
                self.stride,
            ):
                subObject = nextFrame[i : i + self.w, j : j + self.h]
                subObjects.append(subObject)
                indexes.append((i, j))

        values = self.pool.starmap(
            func, zip(subObjects, itertools.repeat(self.selectedObject))
        )
        self.x, self.y = indexes[np.argmin(values)]

        self.selectedObject = subObjects[np.argmin(values)] * 0.75 + self.first * 0.25
        
        
        # plt.imshow(self.selectedObject)
        # plt.show()
        return cv2.rectangle(
            nextFrame, (self.y, self.x), (self.y + self.h, self.x + self.w), (0, 0, 255)
        ), self.selectedObject


def rescale(im: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = im.shape
    if scale == 1.0:
        return im
    return cv2.resize(im, (int(w * scale), int(h * scale)))


if __name__ == "__main__":
    ot, so = None, None
    sf = 1.0
    video = cv2.VideoCapture("video2.mp4")
    
    y = 450
    x = 280
    w = int(640-450)
    h =  int(365 - 280)
    while video.isOpened():
        # videoture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            frame = rescale(frame, sf)
            # plt.imshow(frame[280: 365, 450:640])
            # plt.show()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
            if ot is None:
                ot = ObjectTracker([x,y,w,h], frame)
            else:
                start = time.time()
                frame, so = ot.trackObject(frame)
                end = time.time()
                print(end - start)
               

            # Display the resulting frame
            cv2.imshow("Frame", frame)
            if so is not None:
                cv2.imshow("So", so)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
