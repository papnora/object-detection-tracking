import torch
print(torch.__version__)
print(torch.cuda.is_available())
import sys
import cv2
sys.path.append('/notebooks/ObjectDetectionTracking_PN')
from deepsort_tracking import DeepSORT 
import os 
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

csv_path = '/notebooks/ObjectDetectionTracking_PN/datas/detections/hongkong_pedestrians1.csv'
df = pd.read_csv(csv_path)
print(df)
print("A beolvasás sikeres volt.")

video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/hongkong_pedestrians.mp4'
tracking_video_out = Path('/notebooks/ObjectDetectionTracking_PN/datas/videos/deepsort_out.avi')


tracker = DeepSORT()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
#--

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
frame_id = 0  
width = frame.shape[1]
height = frame.shape[0]
print(frame.shape)
out = cv2.VideoWriter(str(tracking_video_out), cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (width, height))


try: 
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
        
        pred = df[df['frame_id'] == frame_id]
        
        #[print(d) for d in pred.itertuples()]
        tracker.update(list(pred.itertuples()))
        #print(tracker.tracks)
    
        # kirajzolás
        for track in tracker.tracks:
            bbox = track.bbox  # bbox koordinátái - top-left bottom-right / 4 elemű tuple lista
            track_id = track.track_id  

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[track_id % len(colors)], 2)
            cv2.putText(frame, str(track_id), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[track_id % len(colors)], 2)

        out.write(frame)
        frame_id += 1

except KeyboardInterrupt:
    print('Exiting...')
    
cap.release()
out.release()
cv2.destroyAllWindows()



                          
