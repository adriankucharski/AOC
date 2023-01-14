import math
from matplotlib import pyplot as plt
import numpy as np
from skimage import metrics, filters, morphology
import cv2
from sklearn.metrics import jaccard_score
from typing import Any, Tuple, Deque
import pytictoc
from sklearn import tree
from functools import partial
from skimage import data, segmentation, feature, future
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from collections import deque
from skimage import exposure
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier

class SegmentationTracker:
    def __init__(
        self,
        fit_per_call=2,
        n_estimators=25,
        n_jobs=12,
        max_depth=5,
        max_samples=0.1,
        sigma_min=0.5,
        sigma_max=15,
        intensity=True,
        edges=True,
        texture=True,
        preprocess_gamma=0.5,
        dataset_size = 5,
        margin = 30,
        as_circle_mask=False, 
        fit_stride=4
    ) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            max_depth=max_depth,
            max_samples=max_samples,
            criterion="entropy",
            class_weight="balanced_subsample",
        )
        self.features_func = partial(
            feature.multiscale_basic_features,
            intensity=intensity,
            edges=edges,
            texture=texture,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channel_axis=-1,
        )
        self.fit_stride=fit_stride
        self.trainded = self.clf
        self.gamma = preprocess_gamma
        self.fit_counter = 0
        self.fit_per_call = fit_per_call
        self.margin = margin
        self.as_circle_mask = as_circle_mask
        self.training_data: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=dataset_size)

    def __fit(self):
        if self.fit_counter % self.fit_per_call == 0:
            _x, _y = [], []
            for (xi,yi) in self.training_data:
              _x.append(xi), _y.append(yi)
            self.trainded = future.fit_segmenter(np.asarray(_y), np.asarray(_x), self.clf)
        self.fit_counter += 1

    def __predict(self, X: np.ndarray) -> np.ndarray:
        X = exposure.adjust_gamma(X, self.gamma)
        features = self.features_func(X)
        return future.predict_segmenter(features, self.clf)

    def __append_data(self, X: np.ndarray, y: np.ndarray):
        X = exposure.adjust_gamma(X, self.gamma)
        X = self.features_func(X)
        self.training_data.append((X, y))

    def __preprocess(self, im: np.ndarray) -> np.ndarray:
        im = im / 255.0
        return im

    def reset(self):
        self.training_data.clear()
        self.trainded = self.clf
        self.fit_counter = 0

    def get_next_box(self, selected_box: tuple, prev_frame: np.ndarray, next_frame: np.ndarray) -> tuple:
        prev_frame, next_frame = self.__preprocess(prev_frame), self.__preprocess(next_frame)
        height, width = prev_frame.shape[:2]
        y, x, w, h = selected_box
        
        margin_slice_idx = SegmentationTracker.to_slice(selected_box, height, width, self.margin)
        mask = SegmentationTracker.to_mask(selected_box, height, width, self.margin, self.as_circle_mask)
        
        self.prev_element = np.copy(prev_frame[margin_slice_idx])
        self.__append_data(self.prev_element, mask)
        self.__fit()

        self.next_element = np.copy(next_frame[margin_slice_idx])
        self.next_element_mask = self.__predict(self.next_element)

        dy, dx = SegmentationTracker.fit_window(self.next_element_mask, (h, w), stride=self.fit_stride)
        x = min(max(0, x - self.margin) + dx, width - w - self.margin)
        y = min(max(0, y - self.margin) + dy, height - h - self.margin)

        selected_box = (y, x, w, h)

        return selected_box

    @staticmethod
    def fit_window(seg_im: np.ndarray, window_size: tuple, stride: int = 3) -> tuple:
        seg_im == seg_im.max()
        seg_im = morphology.erosion(seg_im)
        
        windows = np.lib.stride_tricks.sliding_window_view(
            seg_im, window_size, axis=(0, 1), writeable=False
        )[::stride, ::stride]

        values = dict()
        for y in range(windows.shape[0]):
            for x in range(windows.shape[1]):
                values[(y * stride, x * stride)] = np.sum(windows[y, x])

        return max(values, key=values.get)

    @staticmethod
    def to_mask(
        box: tuple, height: int, width: int, margin: int, as_circle=False
    ) -> np.ndarray:
        (y, x, w, h) = box
        mx = min(x, width - w)
        my = min(y, height - h)
        arr = np.ones((height, width))

        if as_circle:
            arr = cv2.circle(arr, (mx + w // 2, my + h // 2), (w + h) // 4, 2, -1)
        else:
            arr[my : my + h, mx : mx + w] = 2

        x = min(max(0, x - margin), width - w - margin)
        y = min(max(0, y - margin), height - h - margin)
        index = np.s_[y : y + h + margin * 2, x : x + w + margin * 2]
        return arr[index]

    @staticmethod
    def to_slice(box: tuple, height: int, width: int, margin: int) -> np.s_:
        (y, x, w, h) = box
        x = min(max(0, x - margin), width - w - margin)
        y = min(max(0, y - margin), height - h - margin)
        return np.s_[y : y + h + margin * 2, x : x + w + margin * 2]