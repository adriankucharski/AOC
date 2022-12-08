import cv2

from gui import SelectBoxWindow
from tracker import ObjectTracker

video = cv2.VideoCapture("videos/video2.mp4")
box = SelectBoxWindow.show_and_get_box(video)
tracker = ObjectTracker(video, box, stride=6, margin=30, scale_factor=0.8)
tracker.track()
