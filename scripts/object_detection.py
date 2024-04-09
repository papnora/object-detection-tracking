import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append('/notebooks/ObjectDetectionTracking_PN/yolov7')
from models.experimental import attempt_load  # YOLOv7 modell betöltése
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from torch import nn

yaml_path = '/notebooks/ObjectDetectionTracking_PN/yolov7/data/coco.yaml'
with open(yaml_path, 'r') as yaml_file:
    coco_data = yaml.safe_load(yaml_file)
names = coco_data['names']

def load_weights(weights: str, imgsz: int = 640, batch_size: int = 1, device: str = "cuda") -> nn.Module:
        ckpt = torch.load(weights, map_location=device)
        model = Ensemble().append(ckpt['ema' if ckpt.get(
            'ema') else 'model'].float().fuse().eval())  # FP32 model
        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        return model[-1]

def process_frame(frame, model, stride, names, frame_id, detections):  
    img = letterbox(frame, 640, stride=stride)[0]  # Képkocka átméretezése és padding
    img = img.transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cuda')
    img = img.half()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Detektálás
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalizált gain whwh

    # Detekciók rajzolása és rögzítése
    for i, det in enumerate(pred):  # Detekciók az egyes képkockákhoz
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                class_name = names[int(cls)]
                label_with_index = f'{int(cls)}: {class_name} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label_with_index, color=(255, 0, 0), line_thickness=3)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).cpu().tolist()  # normalizált xywh
                cx, cy, w, h = xywh  # Itt cx,cy a középpont
                # Átalakítjuk, hogy jobb alsó sarok legyen
                x = cx + w / 2
                y = cy + h / 2
                row = {
                    'frame_id': frame_id, 
                    'label': int(cls.cpu()), 
                    'conf': float(conf.cpu().item()), 
                    'x': x, 
                    'y': y, 
                    'w': w, 
                    'h': h
                }
              #  x,y,w,h = xywh
               # row = {'frame_id': frame_id, 'label': int(cls.cpu()), 'conf': float(conf.cpu().item()), 'x': x, 'y': y, 'w': w, 'h': h}
                print(row)
                detections.loc[len(detections)] = row
    return detections


def process_video(video_path, output_path, model, names):    
    cap = cv2.VideoCapture(video_path)
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    detections = pd.DataFrame(columns=['frame_id', 'label', 'conf', 'x', 'y', 'w', 'h'])

    frame_id = 0
    stride = int(model.stride.max())  # modell stride
    model.eval()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = process_frame(frame, model, stride, names, frame_id, detections)
        frame_id += 1

    detections.to_csv(output_path, index=False)
    cap.release()

model_path = '/notebooks/ObjectDetectionTracking_PN/weights/yolov7-tiny.pt'
video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/park_people.mp4'
output_path = '/notebooks/ObjectDetectionTracking_PN/datas/detections/park_people_detections.csv'

model = attempt_load(model_path)  # Modellsúlyok betöltése
model.half()
names = model.module.names if hasattr(model, 'module') else model.names
model.eval()

process_video(video_path, output_path, model, names)




