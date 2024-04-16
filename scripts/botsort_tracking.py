import sys
import os
from abc import ABC, abstractmethod
sys.path.append('/notebooks/ObjectDetectionTracking_PN/BoT-SORT')
from tracker.bot_sort import BoTSORT


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

class BoTSORT(BaseTracker):
    def __init__(self, args):
        self.args = args
        self.tracker = BoTSORT(args)

    def update(self, detections, img):
        output_results = self.format_detections(detections)
        output_stracks = self.tracker.update(output_results, img)
        formatted_tracks = self.format_tracks(output_stracks)
        return formatted_tracks

    def format_detections(self, detections):
        # Assuming `detections` is a list of dictionaries with keys 'bbox' (x1, y1, x2, y2), 'score', and 'class'
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class']
            formatted_detections.append([x1, y1, x2, y2, score, class_id])
        return np.array(formatted_detections)

    def format_tracks(self, stracks):
        # Convert STrack objects into a more simple dictionary format if necessary
        formatted_tracks = []
        for strack in stracks:
            track_info = {
                'track_id': strack.track_id,
                'bbox': strack.tlbr,  # top-left x, top-left y, bottom-right x, bottom-right y
                'score': strack.score
            }
            formatted_tracks.append(track_info)
        return formatted_tracks


if __name__ == "__main__":
    args = {
        'track_high_thresh': 0.6,
        'track_low_thresh': 0.3,
        'new_track_thresh': 0.7,
        'track_buffer': 30,
        'proximity_thresh': 0.2,
        'appearance_thresh': 0.3,
        'with_reid': True,
        'fast_reid_config': 'config/path.yml',
        'fast_reid_weights': 'weights/path.pth',
        'device': 'cuda'
    }
    tracker = BoTSORTTracker(args)
    detections = [{'bbox': (50, 50, 150, 150), 'score': 0.95, 'class': 1}]
    img = np.zeros((480, 640, 3), dtype=np.uint8)  # Example image
    tracks = tracker.update(detections, img)
    print(tracks)