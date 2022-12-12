from multiprocessing.pool import ThreadPool
import multiprocessing
import cv2
import numpy as np
from typing import List, Tuple
import itertools
from collections import deque
from skimage import filters


def get_keypoints(
    img: np.ndarray, nfeatures=40, nOctaveLayers: int = 3, sigma: float = 1.5
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures, nOctaveLayers, sigma=sigma)
    kp, dsc = sift.detectAndCompute(gray, None)
    return kp, dsc, gray


def center_of_mass_match(
    img1: np.ndarray,
    img2: np.ndarray,
    cv2_norm: int = cv2.NORM_L2,
    nfeatures=40,
    nOctaveLayers: int = 3,
    sigma: float = 1.5,
):
    _, dsc1, _ = get_keypoints(img1, nfeatures, nOctaveLayers, sigma)
    kp2, dsc2, _ = get_keypoints(img2, nfeatures, nOctaveLayers, sigma)
    bf = cv2.BFMatcher(cv2_norm, crossCheck=False)
    matches = bf.match(dsc1, dsc2)
    indexes = np.asarray([kp2[m.trainIdx].pt for m in matches], dtype="int")
    x, y = np.mean(indexes, axis=0, dtype="int")
    return y, x


class ObjectTracker:
    def __init__(
        self,
        first_frame: np.ndarray,
        box: Tuple[int, int, int, int],
        stride: int = 4,
        margin: int = 30,
        coords_mem_size: int = 5,
        sigma: float = 2.0,
        first_last_ratio: Tuple[float, float] = [0.4, 0.6],
    ) -> None:
        self.x, self.y, self.h, self.w = box
        self.stride = stride
        self.margin = margin
        self.sigma = sigma
        self.height, self.width = first_frame.shape[:2]
        self.first_last_ratio = first_last_ratio

        first_frame = filters.gaussian(first_frame, sigma=self.sigma, channel_axis=-1)
        self.pool = ThreadPool(multiprocessing.cpu_count())
        self.selected_object = self.first = first_frame[
            self.x : self.x + self.w, self.y : self.y + self.h
        ]
        self.coords_memory = deque(
            [(self.y, self.x) for _ in range(coords_mem_size)], maxlen=coords_mem_size
        )

        weights = np.arange(1, coords_mem_size + 1) ** 2
        self.weights = weights / np.sum(weights)
        self.coords_all = []

    def process_frame(self, next_frame: np.ndarray) -> tuple:
        next_frame = filters.gaussian(next_frame, sigma=self.sigma, channel_axis=-1)
        sub_objects, indexes = self._get_sub_objects_and_indexes(next_frame)

        values = self.pool.starmap(
            self.func, zip(sub_objects, itertools.repeat(self.selected_object))
        )
        best_index = np.argmin(values)
        best_sub_object = sub_objects[best_index]
        x, y = indexes[best_index]

        self.coords_all.append((y, x))

        self.coords_memory.append((y, x))

        if len(self.coords_memory) > 2:
            diff = np.diff(self.coords_memory, axis=0)
            # If last diff is >> than second last diff
            dy1, dy2 = diff[-1, 0], diff[-2, 0]
            dx1, dx2 = diff[-1, 1], diff[-2, 1]
            if abs(dy1 - dy2) > self.margin * 2 or abs(dx1 - dx2) > self.margin * 2:
                self.coords_memory.pop()
                ly, lx = self.coords_memory[-1]
                y, x = ly + dy2, lx + dx2
                height, width = self.first.shape[:2]
                y, x = min(max(0, y), self.height - self.h), min(
                    max(0, x), self.width - self.w
                )
                print(y, x, height, width)
                self.coords_memory.append((y, x))

        avg_y, avg_x = np.average(self.coords_memory, axis=0, weights=self.weights)
        self.y, self.x = int(avg_y), int(avg_x)

        cfirst, clast = self.first_last_ratio
        self.selected_object = best_sub_object * clast + self.first * cfirst
        return self.y, self.x, self.h, self.w

    def _get_sub_objects_and_indexes(self, frame: np.ndarray):
        vw, vh, _ = frame.shape

        sub_objects: List[np.ndarray] = []
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
                sub_object = frame[i : i + self.w, j : j + self.h]
                sub_objects.append(sub_object)
                indexes.append((i, j))

        return sub_objects, indexes

    @staticmethod
    def func(sub_object: np.ndarray, selected_object: np.ndarray) -> float:
        return np.sum((sub_object - selected_object) ** 2)
