import torch
import sys
sys.path.append('/notebooks/ObjectDetectionTracking_PN/yolov7')
sys.path.append('/notebooks/ObjectDetectionTracking_PN/deep_sort/deep_sort')
#from deep_sort.deep_sort import DeepSort
from deep_sort.deep_sort import DeepSort
#from deep_sort.nn_matching import NearestNeighborDistanceMetric #meghatározza a távolságmértéket
#from deep_sort.detection import Detection as DeepSORTDetection
#from deep_sort.tracker import Tracker #távolságmértéket használva inicializálja a Tracker objektumot

import cv2
import numpy as np
from models.experimental import attempt_load  # YOLOv7 model loading
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

# Load YOLOv7 model
model = attempt_load('/notebooks/ObjectDetectionTracking_PN/weights/yolov7-tiny.pt')
model.half()
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names
model.eval()

# Video processing
video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/hongkong_pedestrians.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
width = frame.shape[1]
height = frame.shape[0]
print(frame.shape)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/notebooks/ObjectDetectionTracking_PN/output/processed_video.avi', fourcc, 20.0, (width, height))

# Initialize DeepSORT
deepsort = DeepSort(
    model_path="/notebooks/ObjectDetectionTracking_PN/deepsort/checkpoints/ckpt.t7",
    max_dist=0.4, 
    min_confidence=0.3,
    nms_max_overlap=0.5,
    max_iou_distance=0.7,
    max_age=70, 
    n_init=3,
    nn_budget=100,
    use_cuda=True
)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('End of video')
            break

        # Frame preparation
        img = letterbox(frame, 640, stride=stride)[0]  # Resize and pad the frame
        img = img.transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cuda')
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Detection with YOLOv7
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        # Prepare detections for DeepSORT
        bbox_xywh = []
        confidences = []
        classes = []
        
        for i, det in enumerate(pred):  # Detections for each frame
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                    confidences.append(conf.item())
                    classes.append(cls.item())
        
        xywhs = torch.Tensor(bbox_xywh)
        confs = torch.Tensor(confidences)
        
        # Update DeepSORT tracker
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), classes, frame)
        
        # Draw tracking results
        for output in outputs:
            bbox = output[:4]
            track_id = output[4]
            class_id = output[5]
            label = f'{names[int(class_id)]} {track_id}'
            plot_one_box(bbox, frame, label=label, color=(255, 0, 0), line_thickness=3)
        
        # Output video frame
        out.write(frame)
except KeyboardInterrupt:
    print('Exiting...')

cap.release()
out.release()
cv2.destroyAllWindows()