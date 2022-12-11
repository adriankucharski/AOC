import time
import cv2
import numpy as np
from gui import SelectBoxWindow
from tracker import ObjectTracker
from pytictoc import TicToc
timer = TicToc()

def resize_frame(im: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = im.shape
    if scale == 1.0:
        return im
    return cv2.resize(im, (int(w * scale), int(h * scale)))


def preprocess_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    scaled_frame = np.array(frame / 255.0, dtype='float32')
    resized_frame = resize_frame(scaled_frame, scale)
    return resized_frame


def show_tracking_animation(video: cv2.VideoCapture, tracker: ObjectTracker, scale: float, debug = False):
    while video.isOpened():
        _, frame = video.read()
        frame = preprocess_frame(frame, scale)

        # Print time of frame tracking
        if debug:
            timer.tic()
        new_box = tracker.process_frame(frame)
        if debug:
            timer.toc()
            
        x, y, w, h = new_box

        # Draw found box on frame
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=1)

        # Display the resulting frame
        cv2.imshow("Frame", frame)
        if tracker.selected_object is not None:
            cv2.imshow("Object", tracker.selected_object)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break