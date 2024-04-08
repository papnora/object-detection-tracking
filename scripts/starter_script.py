import os
import random
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.append('/notebooks/ObjectDetectionTracking_PN')
from deepsort_tracking import DeepSORT 
from bytetrack_tracking import ByteTrack

print(torch.__version__)
print(torch.cuda.is_available())

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)
    
def load_dataframe(csv_path):
    return pd.read_csv(csv_path)

def initialize_video_writer(video_path, width, height):
    return cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (width, height))

def main():
    yaml_path = '/notebooks/ObjectDetectionTracking_PN/yolov7/data/coco.yaml'
    coco_data = load_yaml_config(yaml_path)
    names = coco_data['names']

    csv_path = '/notebooks/ObjectDetectionTracking_PN/datas/detections/park_people.csv'
    df = load_dataframe(csv_path)

    video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/park_people.mp4'
    tracking_video_out = Path('/notebooks/ObjectDetectionTracking_PN/datas/videos/bytetrack_out.avi')
    #tracking_video_out = Path('/notebooks/ObjectDetectionTracking_PN/datas/videos/deepsort_out.avi')
    
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame_id = 0  
    width = frame.shape[1]
    height = frame.shape[0]
    tracker = ByteTrack(imsz=(height, width))
    #tracker = DeepSORT()
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

            # kirajzolás
            for track in tracker.tracks:
                bbox = track.bbox  # bbox koordinátái - top-left bottom-right / 4 elemű tuple lista
                track_id = track.track_id  

                #'pred' DataFrame-ben 'label' oszlop - objektum kategória indexe
                label_index = pred[pred['frame_id'] == frame_id]['label'].values[0]  #az első detektált objektum típusa minden frame-ben
                label_name = names[label_index]  


                cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[track_id % len(colors)], 2)
                cv2.putText(frame, label_name, (int(bbox[0]), int(bbox[1]+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[track_id % len(colors)], 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[track_id % len(colors)], 2)


            out.write(frame)
            frame_id += 1

    except KeyboardInterrupt:
        print('Exiting...')

    finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()