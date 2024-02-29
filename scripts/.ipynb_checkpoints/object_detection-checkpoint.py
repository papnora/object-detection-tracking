import torch
import sys
sys.path.append('/notebooks/ObjectDetectionTracking_PN/yolov7')
import cv2 
import numpy as np
from models.experimental import attempt_load  # YOLOv7 modell betöltése
from torch import nn
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

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
# Modell betöltése és előkészítése

model = attempt_load('/notebooks/ObjectDetectionTracking_PN/weights/yolov7-tiny.pt')  # Modellsúlyok elérési útvonala
model.half()
stride = int(model.stride.max())  # modell stride
names = model.module.names if hasattr(model, 'module') else model.names
model.eval()


video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/hongkong_pedestrians.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
width = frame.shape[1]
height = frame.shape[0]
print(frame.shape)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
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

        # Detekciók rajzolása a képkockára
        for i, det in enumerate(pred):  # Detekciók az egyes képkockákhoz
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=3)

        #cv2.imshow('YOLOv7 Object Detection', frame)
        out.write(frame)
        #if cv2.waitKey(1) == ord('q'):  # 'q' billentyűvel kilépés
except KeyboardInterrupt: 
    print('Exiting...')

cap.release()
out.release()
cv2.destroyAllWindows()
