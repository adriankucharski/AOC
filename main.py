import time
import cv2
import numpy as np
from gui import SelectBoxWindow
from tracker import ObjectTracker
from pytictoc import TicToc
import imageio

timer = TicToc()


def resize_frame(im: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = im.shape
    if scale == 1.0:
        return im
    return cv2.resize(im, (int(w * scale), int(h * scale)))


def preprocess_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    scaled_frame = np.array(frame / 255.0, dtype="float32")
    resized_frame = resize_frame(scaled_frame, scale)
    return resized_frame


def show_tracking_animation(
    video: cv2.VideoCapture,
    tracker: ObjectTracker,
    scale: float,
    thickness: int = 1,
    num_of_frames: int = None,
    debug = False,
    save_path: str = None,
    fps: int = 30
):
    i = 0
    if save_path is not None:
        frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret or num_of_frames is not None and i >= num_of_frames:
            break
        i += 1
        frame = preprocess_frame(frame, scale)

        # Print time of frame tracking
        if debug:
            timer.tic()
        new_box = tracker.process_frame(frame)
        if debug:
            timer.toc()

        x, y, w, h = new_box

        
        # Draw found box on frame
        frame = cv2.rectangle(
            frame, (x, y), (x + w, y + h), (0, 0, 1.0), thickness=thickness
        )
        
        # Display the resulting frame
        cv2.imshow("Frame", frame)

        if tracker.selected_object is not None:
            cv2.imshow("Object", tracker.selected_object)

        if save_path is not None:
            frames.append(frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()

    if save_path is not None:
        with imageio.get_writer(save_path, mode="I", fps=fps, codec="libx264") as writer:
            for frame in frames:
                valid = np.array(frame[..., ::-1] * 255, dtype='uint8')
                writer.append_data(valid)
