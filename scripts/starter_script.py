import os
import sys
import time
import random
from pathlib import Path
from functools import wraps


import cv2
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from sklearn.model_selection import ParameterGrid

sys.path.append('/notebooks/ObjectDetectionTracking_PN')
from deepsort_tracking import DeepSORT 
from bytetrack_tracking import ByteTrack
from botsort_tracking import BoTSORTTracker

print(torch.__version__)
print(torch.cuda.is_available())


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)
    
def load_dataframe(csv_path_detection):
    return pd.read_csv(csv_path_detection)

def save_tracking_to_csv(tracking_data, csv_path_tracking_output, video_path, tracker_type):
    csv_path_tracking_output = Path(csv_path_tracking_output)
    if not csv_path_tracking_output.exists():
        csv_path_tracking_output.mkdir(parents=True)
        
    video_name = Path(video_path).stem
    base_filename = f"{video_name}_{tracker_type}"
    output_path = csv_path_tracking_output / (f"{base_filename}.csv")

    if output_path.exists():
        existing_files = list(csv_path_tracking_output.glob(f'{base_filename}_*.csv'))
        file_count = len(existing_files)
        output_path = csv_path_tracking_output / (f"{base_filename}_{file_count}.csv")
    
    tracking_data.to_csv(output_path, index=False)
    print(f"Tracking data saved to: {output_path}")


def initialize_video_writer(video_path, width, height):
    return cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (width, height))

def performance_monitor(func):
    results = pd.DataFrame(columns=['Objektumkövető', 'Execution Time (s)', 'Memory Usage (MiB)'])
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker_type = args[0] 
        start_time = time.perf_counter()
        mem_usage_start = memory_usage()[0]
        result = func(*args, **kwargs)  # Run the function
        mem_usage_end = memory_usage()[0]
        elapsed_time = time.perf_counter() - start_time
        memory_used = mem_usage_end - mem_usage_start

        print(f"Tracking completed in {elapsed_time:.4f} seconds.")
        print(f"Memory used: {mem_usage_end - mem_usage_start} MiB")
        
        # Store results
        results.loc[len(results)] = [tracker_type,  elapsed_time, memory_used]
        
        if 'save_performance_data' in kwargs and kwargs['save_performance_data']:
            save_performance_to_excel(results, kwargs.get('metrics_path', './'))

        return result
    
    return wrapper

def save_performance_to_excel(df, path):
   # Ensure the directory exists
    Path(path).mkdir(parents=True, exist_ok=True)
    # Save the DataFrame to an Excel file
    excel_path = os.path.join(path, "resource_performance_data.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Performance data saved to Excel at: {excel_path}")

@performance_monitor
def run_tracking(tracker_type, video_path, tracking_video_out, df, names, csv_path_tracking_output, **kwargs):
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
    tracking_data = pd.DataFrame(columns=['frame_id', 'id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'presence'])
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame_id = 0 
    width = frame.shape[1]
    height = frame.shape[0]

    if tracker_type == 'botsort':
        tracker = BoTSORTTracker()
    elif tracker_type == 'bytetrack':
        tracker = ByteTrack(imsz=(height, width))
    elif tracker_type == 'deepsort':
        tracker = DeepSORT()
    
    else:
        raise ValueError("Unsupported tracker type")
        
    print(f"\nRunning {tracker_type} tracking...\n")

    out = initialize_video_writer(tracking_video_out, width, height)
    
    try:
        for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f'Frame: {frame_id}', (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            
            pred = df[df['frame_id'] == frame_id]
            tracker.update(list(pred.itertuples()))

            for track in tracker.tracks:
                bbox = track.bbox
                track_id = track.track_id
                label_index = pred[pred['frame_id'] == frame_id]['label'].values[0]
                label_name = names[label_index]

                cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[track_id % len(colors)], 2)
                cv2.putText(frame, label_name, (int(bbox[0]), int(bbox[1]+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[track_id % len(colors)], 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[track_id % len(colors)], 2)
                
                # adat a csv kiírásra
                tracking_data.loc[len(tracking_data)] = [frame_id+1, track_id, int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1]), 1] 

            out.write(frame)
            frame_id += 1

    except KeyboardInterrupt:
        print('Exiting...')

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        save_tracking_to_csv(tracking_data, csv_path_tracking_output, video_path, tracker_type)

def main():
    yaml_path = '/notebooks/ObjectDetectionTracking_PN/yolov7/data/coco.yaml'
    coco_data = load_yaml_config(yaml_path)
    names = coco_data['names']

    csv_path_detection = '/notebooks/ObjectDetectionTracking_PN/datas/detections/park_people.csv'
    df = load_dataframe(csv_path_detection)
    
    csv_path_tracking_output = '/notebooks/ObjectDetectionTracking_PN/datas/trackings/'

    video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/park_people.mp4'
    metrics_path = '/notebooks/ObjectDetectionTracking_PN/datas/metrics/'

    tracking_video_out_byte = Path('/notebooks/ObjectDetectionTracking_PN/datas/videos/bytetrack_out.avi')
    run_tracking('bytetrack', video_path, tracking_video_out_byte, df, names, csv_path_tracking_output, save_performance_data=True, metrics_path=metrics_path)
        
    tracking_video_out_deep = Path('/notebooks/ObjectDetectionTracking_PN/datas/videos/deepsort_out.avi')
    run_tracking('deepsort', video_path, tracking_video_out_deep, df, names, csv_path_tracking_output, save_performance_data=True, metrics_path=metrics_path)
    
    tracking_video_out_bot = Path('/notebooks/ObjectDetectionTracking_PN/datas/videos/botsort_out.avi')
    run_tracking('botsort', video_path, tracking_video_out_bot, df, names, csv_path_tracking_output, save_performance_data=True, metrics_path=metrics_path)

if __name__ == "__main__":
    main()