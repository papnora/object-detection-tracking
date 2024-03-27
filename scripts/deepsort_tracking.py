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
        max_cosine_distance = 0.3 #0.4 volt
        nn_budget = 100 #100 volt
        max_age = 90 # elfelejtési idő - tracker fenntartja a tracket 90 képkockán keresztül, miután az objektumot már nem észlelte - 20fps
        n_init = 3 #hány megerősített detektálás szükséges egy track létrehozásához
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.tracks = []

    def update(self, detections): 
        self.tracker.predict()
        deepsort_detections = [Detection([(d.x - d.w), (d.y - d.h), d.w, d.h], d.conf, []) for d in detections]
        #deepsort_detections = [Detection([(d.x - (d.w / 2)), (d.y - (d.h / 2)), (d.x + (d.w / 2)), (d.y + (d.h / 2))], d.conf, []) for d in detections]
        self.tracker.update(deepsort_detections)
        self.update_tracks()        

    def update_tracks(self):
        self.tracks = [Track(track.track_id, track.to_tlbr()) for track in self.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]

