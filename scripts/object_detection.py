import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append('/notebooks/ObjectDetectionTracking_PN/yolov7')
from models.experimental import attempt_load  
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


def process_video(video_path, model_weights, output_dir):
        model = attempt_load(model_weights)  
        model.half()
        stride = int(model.stride.max())  # modell stride
        names = model.module.names if hasattr(model, 'module') else model.names
        print(names)
        model.eval()  
        
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        output_path = output_dir / (Path(video_path).with_suffix(".csv").name)
        if output_path.exists():
            fs = [f for f in output_dir.glob(f'*{output_path.stem}*')]
            output_path = output_dir / (str(Path(video_path).stem) + f'{len(fs)}.csv')
                                    
        detections = pd.DataFrame(columns=['frame_id','label', 'conf', 'x', 'y', 'w', 'h']) # x,y bounding box kp koodrinátái - w,h bounding box szélessége és magassága,  conf - confidencia, label  + cls az objektum osztályaz. (string , szám)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print("Nem sikerült megnyitni a videót.")
            return
        frame_id = 0
        width = frame.shape[1]
        height = frame.shape[0]
        print(frame.shape)
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
                                    
        try: 
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print('Videó vége')
                    break

                # Képkocka előkészítése
                img = letterbox(frame, 640, stride=stride)[0]  # Képkocka átméretezése és padding
                img = img.transpose(2, 0, 1)  # BGR to RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to('cuda')
                img = img.half()

                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Detektálás
                with torch.no_grad():  #gradiensek számítása GPU memóriaszivárgást okozna
                    pred = model(img, augment=False)[0]
                    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

                gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalizált gain whwh

                # Detekciók rajzolása a képkockára
                for i, det in enumerate(pred):  # detekciók az egyes képkockákhoz
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in det:
                            class_name = names[int(cls)]
                            label_with_index = f'{int(cls)}: {class_name} {conf:.2f}'
                            plot_one_box(xyxy, frame, label=label_with_index, color=(255, 0, 0), line_thickness=3)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).cpu().tolist()  # normalizált xywh
                            cx, cy, w, h = xywh  #cx,cy a középpont
                            # átalakítom jobb alsó sarokká
                            x = cx + w / 2
                            y = cy + h / 2
                            row = {
                                'frame_id': frame_id, 
                                'label': int(cls.cpu()), #'label': names[int(cls.cpu())]
                                'conf': float(conf.cpu().item()), 
                                'x': x, 
                                'y': y, 
                                'w': w, 
                                'h': h
                            }
                            #print(row)
                            detections.loc[len(detections)] = row
                            print("detections")
                            print(detections)

                out.write(frame)
                frame_id += 1 
                    
        except KeyboardInterrupt:
            
            print('Exiting...')
        finally:
            detections.to_csv(output_path, index=False)
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"A detekciók mentve: {output_path}")

                                                                        
video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/park_people.mp4'
model_weights = '/notebooks/ObjectDetectionTracking_PN/weights/yolov7-tiny.pt'
output_dir = '/notebooks/ObjectDetectionTracking_PN/datas/detections/'

process_video(video_path, model_weights, output_dir)