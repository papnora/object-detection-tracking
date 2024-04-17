import sys
import os
import numpy as np
import argparse
from abc import ABC, abstractmethod

sys.path.append('/notebooks/ObjectDetectionTracking_PN/BoT-SORT')
from tracker.bot_sort import BoTSORT, STrack

class BaseTrack(ABC):
   # count = 0 
    
    @abstractmethod
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
    
    #@classmethod
    #def clear_count(cls):
       # cls.count = 0

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
    def __init__(self, track_id, bbox):
        super().__init__(track_id, bbox)

class BoTSORTTracker(BaseTracker):
    def __init__(self,track_high_thresh=0.6, track_low_thresh=0.3, track_buffer=40, match_thresh=0.2, img=(1920, 1080)):
      #  parser = argparse.ArgumentParser()
      #  parser.add_argument("--track_high_thresh", type=float, default=0.6, help="High tracking confidence threshold")
      #  parser.add_argument("--track_low_thresh", type=float, default=0.3, help="Low tracking confidence threshold")
      #  parser.add_argument("--new_track_thresh", type=float, default=0.7, help="New track creation confidence threshold")
      #  parser.add_argument("--track_buffer", type=int, default=30, help="The frames for keeping lost tracks")
      #  parser.add_argument("--proximity_thresh", type=float, default=0.2, help="Proximity threshold for tracking")
      #  parser.add_argument("--appearance_thresh", type=float, default=0.3, help="Appearance threshold for re-identification")
      #  parser.add_argument("--with_reid", action='store_true', help="Flag to enable/disable re-identification")
      #  parser.add_argument("--fast_reid_config", type=str, default='config/path.yml', help="Path to the FastReID config file")
      #  parser.add_argument("--fast_reid_weights", type=str, default='weights/path.pth', help="Path to the FastReID weights")
      #  parser.add_argument("--device", type=str, default='cuda', help="Device to run the tracking model on")
      #  self.tracker = BoTSORT(["--track_high_thresh", str(track_high_thresh), "--track_low_thresh", str(track_low_thresh), "--track_buffer", str(track_buffer)])
       # self.tracks = []
        args = argparse.Namespace(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=0.7,  
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                proximity_thresh=0.2,
                appearance_thresh=0.3,
                with_reid=False,
                fast_reid_config='config/path.yml',
                fast_reid_weights='weights/path.pth',
                device='cuda',
                cmc_method='none',  # Default value for the cmc_method argument
                name='CameraMotionCorrection',  
                ablation=False,  # Default value for the ablation argument
                mot20=False
            
        )
        self.tracker = BoTSORT(args)
        self.tracks = []
        self.img = img

    def update(self, detections):
        output_results = self.format_detections(detections)
        output_stracks = self.tracker.update(output_results, self.img)
        return self.format_tracks(output_stracks)

    def format_detections(self, detections):
        formatted_detections = []
        for det in detections:
            # A itertuples az indexet adja vissza az első elemként, ezért az index + 1-től kezdünk értékeket kinyerni
            frame_id = det[1]
            label = det[2]
            score = det[3]
            x = det[4]
            y = det[5]
            w = det[6]
            h = det[7]

            # Kiszámítjuk a bbox koordinátákat
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            # Hozzáadjuk a formázott detekciókat a listához
            formatted_detections.append([x1, y1, x2, y2, score, label])
        return np.array(formatted_detections)

    def format_tracks(self, stracks):
        formatted_tracks = []
        for strack in stracks:
            track_info = {
                'track_id': strack.track_id,
                'bbox': strack.tlbr,
                'score': strack.score
            }
            formatted_tracks.append(track_info)
        return formatted_tracks

if __name__ == "__main__":
    tracker = BoTSORTTracker()
    detections = [{'bbox': (50, 50, 150, 150), 'score': 0.95, 'class': 1}]
    img = np.zeros((480, 640, 3), dtype=np.uint8)  # Példa kép
    tracks = tracker.update(detections, img)
    print(tracks)