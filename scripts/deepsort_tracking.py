import sys
from abc import ABC, abstractmethod
sys.path.append('/notebooks/ObjectDetectionTracking_PN')
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric  
from deep_sort.deep_sort.detection import Detection
import numpy as np


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


class DeepSORT(BaseTracker):
    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.tracks = []

    def update(self, detections): 
        self.tracker.predict()
        deepsort_detections = [Detection([(d.x - d.w), (d.y - d.h), d.w, d.h], d.conf, []) for d in detections]
        self.tracker.update(deepsort_detections)
        self.update_tracks()        

    def update_tracks(self):
        self.tracks = [Track(track.track_id, track.to_tlbr()) for track in self.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]

