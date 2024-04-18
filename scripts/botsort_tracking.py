import sys
import os
import numpy as np
import argparse
from abc import ABC, abstractmethod

sys.path.append('/notebooks/ObjectDetectionTracking_PN/BoT-SORT')
from tracker.bot_sort import BoTSORT, STrack

class BaseTrack(ABC):
    
    @abstractmethod
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox

class BaseTracker(ABC):
    tracker = None
    tracks = []

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, detections, img=None):
        pass

class Track(BaseTrack):
    def __init__(self, track_id, bbox, score=None):
        super().__init__(track_id, bbox)
        self.bbox = bbox  # A bbox most egy (x1, y1, x2, y2) tuple
        self.score = score

class Detection:
    def __init__(self, frame_id, label, conf, x, y, w, h):
        self.frame_id = frame_id
        self.label = label
        self.conf = conf
        #self.bbox = (x, y, w, h)  # x, y bal felső sarok
        self.bbox = (x - w, y - h, w, h) #x y bal felső lett
        
class BoTSORTTracker(BaseTracker):
    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.2, track_buffer=50, match_thresh=0.2, img=(1920, 1080)):
        args = argparse.Namespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=0.8,  
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            proximity_thresh=0.2,
            appearance_thresh=0.3,
            with_reid=False,
            fast_reid_config='config/path.yml',
            fast_reid_weights='weights/path.pth',
            device='cuda',
            cmc_method='none',
            name='CameraMotionCorrection',
            ablation=False,
            mot20=False
        )
        self.tracker = BoTSORT(args)
        self.tracks = []
        self.img = img

    def update(self, detections):
        output_results = self.format_detections(detections)
        output_results = np.array(output_results, dtype=np.float32)  
        output_stracks = self.tracker.update(output_results, self.img)
        self.tracks = [Track(strack.track_id, strack.tlbr, strack.score) for strack in output_stracks]

    def format_detections(self, detections):
        formatted_detections = []
        for det in detections:
            det = Detection(*det[1:])  # A 'det' itt egy namedtuple, index 0 a pandas index- nem kell
            x1, y1, w, h = det.bbox
            x2 = x1 + w  #  BoTSORT x2, y2 koordinátákat vár --> bbox konvertálás
            y2 = y1 + h
            formatted_detections.append([x1, y1, x2, y2, det.conf])
        return formatted_detections