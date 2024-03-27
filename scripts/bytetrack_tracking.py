import torch
print(torch.__version__)
print(torch.cuda.is_available())
import sys
import cv2
import yaml
sys.path.append('/notebooks/ObjectDetectionTracking_PN/ByteTrack/yolox')
#/ByteTrack/yolox
from tracker.kalman_filter import KalmanFilter
from tracker.basetrack import BaseTrack, TrackState

from tracker.byte_tracker import BYTETracker as ByteTrackTracker
import os 
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import pandas as pd

class BaseTrack(ABC):
    @abstractmethod
    def __init__(self, track_id, bbox):
        pass

class BaseTracker(ABC):
    tracker = None
    tracks = []

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, detections):
        pass

class Track(BaseTrack):
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox

class ByteTrack(BaseTracker):
    def __init__(self, track_thresh=0.5, track_buffer=30):
        self.tracker = BYTETracker({'track_thresh': track_thresh, 'track_buffer': track_buffer}, frame_rate=30)
        self.tracks = []

    def update(self, detections):
        frame_id, label, conf, x, y, w, h = zip(*detections) # CSV-ből olvasott adatok feltételezése
        bboxes = np.array([x, y, w, h]).T
        scores = np.array(conf) # A konfidencia értékek
        outputs = self.tracker.update(bboxes, scores, frame_id[0])

        #`tracks` lista frissítése
        self.tracks = [Track(track.track_id, track.to_tlbr()) for track in outputs]



