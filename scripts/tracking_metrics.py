import motmetrics as mm
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append('/notebooks/ObjectDetectionTracking_PN')

ground_truth_columns = ['Frame', 'ID', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class', 'Visibility']
tracking_columns = ['frame_id', 'id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'presence']

def load_ground_truth(gt_file_path: str) -> pd.DataFrame:
    all_gt_data = []
    try:
        with open(gt_file_path) as f:
            for line in f:
                data_parts = line.strip().split(',')
                try:
                    # Attempt to convert all entries to float, but keep strings as is for non-convertible values
                    frame_data = [float(x) if i != 7 else x for i, x in enumerate(data_parts)]
                    all_gt_data.append(frame_data)
                except ValueError as e:
                    print(f"Skipping line due to error: {e}. Line content: {line}")
    except FileNotFoundError:
        print(f"File not found: {gt_file_path}")
        return pd.DataFrame(columns=ground_truth_columns)
    
    gt_df = pd.DataFrame(all_gt_data, columns=ground_truth_columns)
    print("First few rows of the loaded ground truth data:")
    print(gt_df.head())
    return gt_df

def load_tracking_data(trackings_csv_path: str) -> pd.DataFrame:
    try:
        tracking_df = pd.read_csv(trackings_csv_path, names=tracking_columns)
        print("First few rows of the loaded tracking data:")
        print(tracking_df.head())
    except FileNotFoundError:
        print(f"File not found: {trackings_csv_path}")
        return pd.DataFrame(columns=tracking_columns)
    return tracking_df

def compute_motmetrics(gt_df: pd.DataFrame, test_df: pd.DataFrame) -> mm.MOTAccumulator: 
    acc = mm.MOTAccumulator(auto_id=True)
    
    frames = np.union1d(gt_df['Frame'].unique(), test_df['frame_id'].unique())
    for frame in frames:
        gt_indices = gt_df[gt_df['Frame'] == frame]['ID'].values
        test_indices = test_df[test_df['frame_id'] == frame]['id'].values
        
        gt_boxes = gt_df[gt_df['Frame'] == frame][['X', 'Y', 'Width', 'Height']].values
        test_boxes = test_df[test_df['frame_id'] == frame][['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values

        iou_matrix = mm.distances.iou_matrix(gt_boxes, test_boxes, max_iou=0.5)
        
        acc.update(
            gt_indices,
            test_indices,
            iou_matrix
        )
        
    return acc

def eval_results(trackings_csv_path: str, ground_truth_file: str) -> float:
    gt_df = load_ground_truth(ground_truth_file)
    test_df = load_tracking_data(trackings_csv_path)
    
    acc = compute_motmetrics(gt_df, test_df)

    # TODO: Calculate metrics using motmetrics library
    # mh = mm.metrics.create()
    # summary = mh.compute(acc, metrics=['mota', 'motp'], name='acc')
    # return summary

    # Dummy return, to be replaced by actual metric calculations
    return acc

# Testing
ground_truth_file = Path('/notebooks/ObjectDetectionTracking_PN/datas/ground_truth/gt.txt')
#trackings_csv_path = Path('/notebooks/ObjectDetectionTracking_PN/datas/trackings/park_people_bytetrack.csv')
eval_results(trackings_csv_path ,ground_truth_file)
