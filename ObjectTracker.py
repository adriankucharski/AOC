from multiprocessing.pool import ThreadPool
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple
import cv2
import itertools
import numba
import time
from scipy.ndimage import measurements
from sklearn.metrics import mutual_info_score
from skimage import metrics
from pytictoc import TicToc

from utils import rescale

timer = TicToc()

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def func(subObject: np.ndarray, selectedObject: np.ndarray) -> float:
    # return PSNR(subObject, selectedObject)
    return np.sum((subObject - selectedObject) ** 2)


class ObjectTracker:
    def __init__(
        self,
        video: cv2.VideoCapture,
        box: Tuple[int, int, int, int],
        stride: int = 4,
        margin: int = 30,
        scale_factor: float = 2.5,
        frames_memory: int = 15,
        debug = False
    ) -> None:
        self.video = video
        self.x, self.y, self.h, self.w = [int(b * scale_factor) for b in box]
        self.scale_factor = scale_factor
        self.stride = stride
        self.margin = margin
        self.frames = []
        self.frames_memory = frames_memory
        self.debug = debug

        if video.isOpened():
            ret, frame = video.read()
            if not ret:
                raise Exception("Video stream error")
            frame = self._preprocess_frame(frame)
            self.selectedObject = frame[self.x: self.x +
                                        self.w, self.y: self.y + self.h]
            self.first = np.array(self.selectedObject)

        self.pool = ThreadPool(multiprocessing.cpu_count())

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.array(frame / 255.0, dtype='float32')
        frame = rescale(frame, self.scale_factor)
        return frame

    def _trackObject(self, nextFrame: np.ndarray) -> np.ndarray:
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
                subObject = nextFrame[i: i + self.w, j: j + self.h]
                subObjects.append(subObject)
                indexes.append((i, j))

        values = self.pool.starmap(
            func, zip(subObjects, itertools.repeat(self.selectedObject))
        )
        self.x, self.y = indexes[np.argmin(values)]

        self.frames.append(subObjects[np.argmin(values)])
        last_frames = self.frames[0:5:self.frames_memory]
        last_frames = np.sum(last_frames, axis=0) * 0.25 / len(last_frames)
        
        self.selectedObject = subObjects[np.argmin(
            values)] * 0.75 + self.first * 0.25  # + last_frames
        # self.selectedObject = np.sum(_frames, axis=0) / len(_frames)

        if len(self.frames) > self.frames_memory:
            self.frames.pop(0)

        return cv2.rectangle(
            nextFrame, (self.y, self.x), (self.y + self.h, self.x + self.w), (0, 0, 255), thickness=2
        ), self.selectedObject

    def track(self):
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            frame = self._preprocess_frame(frame)

            start = time.time()
            frame, so = self._trackObject(frame)
            end = time.time()
            print(end - start)

            # Display the resulting frame
            cv2.imshow("Frame", frame)
            if so is not None:
                cv2.imshow("So", so)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
