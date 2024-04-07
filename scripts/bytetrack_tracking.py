from abc import ABC, abstractmethod
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
from yolox.tracker.byte_tracker import BYTETracker #as ByteTrackTracker
import os 
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse

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
    def __init__(self, track_thresh=0.5, track_buffer=30, imsz=(1920, 1080)):
        parser = argparse.ArgumentParser()
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument("--mot20", action="store_true", default=False)
        args = parser.parse_args(["--track_thresh", str(track_thresh), "--track_buffer", str(track_buffer)]) 
        self.tracker = BYTETracker(args, frame_rate=30)
        self.tracks = []
        self.imsz = imsz

    def update(self, detections):            
        output_results = []
        def get_scores_and_bboxes(detection):
            output_results.append(
                [detection.conf, 
                 detection.x-detection.w,detection.y-detection.h,
                 detection.x,detection.y]
            )
        [get_scores_and_bboxes(d) for d in detections]
        print(output_results)
        output_results = np.array(output_results)
        #print(output_results)
        outputs = self.tracker.update(output_results, self.imsz, self.imsz)

        #`tracks` lista frissítése
        self.tracks = [Track(track.track_id, track.tlbr) for track in outputs]
        #[print(t.bbox) for t in self.tracks]



