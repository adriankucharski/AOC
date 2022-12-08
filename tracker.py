from multiprocessing.pool import ThreadPool
import multiprocessing
import numpy as np
from typing import List, Tuple
import itertools


class ObjectTracker:
    def __init__(
        self,
        first_frame,
        box: Tuple[int, int, int, int],
        stride: int = 4,
        margin: int = 30,
        frames_memory: int = 15
    ) -> None:
        self.x, self.y, self.h, self.w = box
        self.stride = stride
        self.margin = margin
        self.frames_memory = frames_memory

        self.frames = []
        self.pool = ThreadPool(multiprocessing.cpu_count())
        self.selected_object = self.first = first_frame[self.x:self.x+self.w, self.y:self.y+self.h]

    def process_frame(self, next_frame: np.ndarray) -> tuple:
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
                sub_object = frame[i: i + self.w, j: j + self.h]
                sub_objects.append(sub_object)
                indexes.append((i, j))

        return sub_objects, indexes

    @staticmethod
    def func(sub_object: np.ndarray, selected_object: np.ndarray) -> float:
        return np.sum((sub_object - selected_object) ** 2)
