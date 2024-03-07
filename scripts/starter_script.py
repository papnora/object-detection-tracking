import torch
print(torch.__version__)
print(torch.cuda.is_available())
import sys
sys.path.append('/notebooks/ObjectDetectionTracking_PN')
from deepsort_tracking import Tracker 
import os 
import random
import pandas as pd
from pathlib import Path

csv_path = '/notebooks/ObjectDetectionTracking_PN/datas/detections/hongkong_pedestrians.csv'
df = pd.read_csv(csv_path)
print(df)
print("A beolvasás sikeres volt.")

video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/hongkong_pedestrians.mp4'
tracking_video_out = Path('/notebooks/ObjectDetectionTracking_PN/datas/detections/deepsort_out.mp4')

#tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
#--
                          
