from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from tracker_seg import SegmentationTracker
import cv2

def load_labels(labels_path: str) -> List[Tuple[int, int, int, int]]:
    with open(labels_path) as f:
        labels = []
        try:
            for line in f.readlines():
                if ',' in line:
                    _x, _y, _w, _h  = [int(num) for num in line.split(',')]
                    labels.append((_y, _x, _w, _h))
                else:
                    _x, _y, _w, _h  = [int(num) for num in line.split(' ')[1:5]]
                    labels.append((_y, _x, _w - _x, _h - _y))
        except:
            print(line)
        return labels

def IoU(box1, box2):
    # Calculate the coordinates of the intersection box
    x1 = max(box1[1], box2[1])
    y1 = max(box1[0], box2[0])
    x2 = min(box1[1] + box1[3], box2[1] + box2[3])
    y2 = min(box1[0] + box1[2], box2[0] + box2[2])
    
    # Calculate the area of the intersection box
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the area of the union
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    
    # Calculate the IoU
    iou = intersection / union
    
    return iou

def video_benchmark(vpath: str, labels_path: str, tracker: SegmentationTracker, show = True, frames: Tuple[int, int] = None):
    ious = []
    video = cv2.VideoCapture(vpath)
    labels = load_labels(labels_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 30.0:
        labels = labels[::8]
        
    if frames is None:
        frames = (1, len(labels))
    else:
        frames = (max(1, frames[0]), min(len(labels), frames[1]))

    print(frames)

    # Skip n frames
    for _ in range((frames[0] - 1)):
        video.read()

    # Get first frame
    pred_box = labels[frames[0] - 1]
    _, prev_frame = video.read()
    
    for i in tqdm(range(*frames)):
        ret, next_frame = video.read()
        if ret == False:
            break
        pred_box = tracker.get_next_box(pred_box, prev_frame, next_frame)
        prev_frame = next_frame

        iou = IoU(pred_box, labels[i])
        ious.append(iou)

        if show: 
            (y,x,w,h) = pred_box
            img = cv2.rectangle(np.copy(next_frame), (x, y), (x + w, y + h), (0, 255, 0), 2)
            (y,x,w,h) = labels[i]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow("img2", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
            
    cv2.destroyAllWindows()
    return np.mean(np.asarray(ious) > 0.5), np.mean(np.asarray(ious) > 0.25), np.mean(ious)