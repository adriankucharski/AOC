import time
import cv2
import numpy as np
from gui import SelectBoxWindow
from tracker import ObjectTracker


def resize_frame(im: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = im.shape
    if scale == 1.0:
        return im
    return cv2.resize(im, (int(w * scale), int(h * scale)))


def preprocess_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    scaled_frame = np.array(frame / 255.0, dtype='float32')
    resized_frame = resize_frame(scaled_frame, scale)
    return resized_frame


def show_tracking_animation(video: cv2.VideoCapture, tracker: ObjectTracker, scale: float):
    while video.isOpened():
        _, frame = video.read()
        frame = preprocess_frame(frame, scale)

        # Print time of frame tracking
        start = time.time()
        new_box = tracker.process_frame(frame)
        x, y, w, h = new_box
        end = time.time()
        print(end - start)

        # Draw found box on frame
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=1)

        # Display the resulting frame
        cv2.imshow("Frame", frame)
        if tracker.selected_object is not None:
            cv2.imshow("Object", tracker.selected_object)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    scale = 1
    video = cv2.VideoCapture("videos/video1.mp4")

    _, first_frame = video.read()
    first_frame = preprocess_frame(first_frame, scale=scale)
    selected_box = SelectBoxWindow.show_and_get_box(first_frame)
    tracker = ObjectTracker(first_frame, selected_box, stride=6, margin=30)
    show_tracking_animation(video, tracker, scale=scale)
