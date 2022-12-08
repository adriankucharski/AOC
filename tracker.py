from multiprocessing.pool import ThreadPool
import multiprocessing
import numpy as np
from typing import List, Tuple
import cv2
import itertools
import numba
import time
from utils import rescale


class ObjectTracker:
    def __init__(
        self,
        video: cv2.VideoCapture,
        box: Tuple[int, int, int, int],
        stride: int = 4,
        margin: int = 30,
        scale_factor: float = 2.5,
        frames_memory: int = 15
    ) -> None:
        self.video = video
        self.x, self.y, self.h, self.w = [int(b * scale_factor) for b in box]
        self.scale_factor = scale_factor
        self.stride = stride
        self.margin = margin
        self.frames = []
        self.frames_memory = frames_memory
        self.pool = ThreadPool(multiprocessing.cpu_count())
        self.selected_object = self.first = self._get_first_frame()

    def _get_first_frame(self):
        _, frame = self.video.read()
        frame = self._preprocess_frame(frame)
        return frame[self.x:self.x+self.w, self.y:self.y+self.h]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.array(frame / 255.0, dtype='float32')
        frame = rescale(frame, self.scale_factor)
        return frame

    def _track_object(self, next_frame: np.ndarray) -> np.ndarray:
        sub_objects, indexes = self._get_sub_objects_and_indexes(next_frame)

        values = self.pool.starmap(
            self.func, zip(sub_objects, itertools.repeat(self.selected_object))
        )
        best_index = np.argmin(values)
        best_sub_object = sub_objects[best_index]
        self.x, self.y = indexes[best_index]

        self.frames.append(best_sub_object)
        last_frames = self.frames[0:5:self.frames_memory]
        last_frames = np.sum(last_frames, axis=0) * 0.25 / len(last_frames)
        if len(self.frames) > self.frames_memory:
            self.frames.pop(0)
        
        self.selected_object = best_sub_object * 0.5 + self.first * 0.25 + last_frames

        return cv2.rectangle(
            next_frame, (self.y, self.x), (self.y + self.h, self.x + self.w), (0, 0, 255)
        ), self.selected_object

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
                sub_object = frame[i: i + self.w, j: j + self.h]
                sub_objects.append(sub_object)
                indexes.append((i, j))

        return sub_objects, indexes

    @staticmethod
    def func(sub_object: np.ndarray, selected_object: np.ndarray) -> float:
        return np.sum((sub_object - selected_object) ** 2)

    def track(self):
        while self.video.isOpened():
            _, frame = self.video.read()
            frame = self._preprocess_frame(frame)

            # Print time of frame tracking
            start = time.time()
            frame, selected_object = self._track_object(frame)
            end = time.time()
            print(end - start)

            # Display the resulting frame
            cv2.imshow("Frame", frame)
            if selected_object is not None:
                cv2.imshow("Object", selected_object)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
