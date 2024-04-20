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
                if len(data_parts) < len(ground_truth_columns):  # Handle incomplete rows
                    data_parts.extend(['1', '1.0'] * (len(ground_truth_columns) - len(data_parts)))
                try:
                    # Convert all to float except for the specific string fields
                    frame_data = [float(x) if i not in [7, 8] else x for i, x in enumerate(data_parts)]
                    all_gt_data.append(frame_data)
                except ValueError as e:
                    print(f"Skipping line due to error: {e}. Line content: {line}")
    except FileNotFoundError:
        print(f"File not found: {gt_file_path}")
        return pd.DataFrame(columns=ground_truth_columns)
    
    gt_df = pd.DataFrame(all_gt_data, columns=ground_truth_columns)
    # Explicitly convert 'Frame' to float or int
    gt_df['Frame'] = gt_df['Frame'].astype(int)
    print("First few rows of the loaded ground truth data:")
   # print(gt_df.head())
    return gt_df

def load_tracking_data(trackings_csv_path: str) -> pd.DataFrame:
    try:
        # read csv assuming the first row is head
        tracking_df = pd.read_csv(trackings_csv_path)
        #convert the 'frame_id' column to an integer for consistent processing
        tracking_df['frame_id'] = tracking_df['frame_id'].astype(int)
        print("First few rows of the loaded trackings data:")
        print(tracking_df.head())
    except FileNotFoundError:
        print(f"File not found: {trackings_csv_path}")
        return pd.DataFrame(columns=tracking_columns)
    except Exception as e:
        print(f"Error processing the file: {e}")
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

def eval_results_for_all_trackers(trackings_csv_path: Path, ground_truth_file: Path):
    gt_df = load_ground_truth(str(ground_truth_file))
    results = {}
    
    for tracking_csv in trackings_csv_path.glob('*.csv'):
        print(f"Evaluating {tracking_csv.name}")
        test_df = load_tracking_data(str(tracking_csv))
        acc = compute_motmetrics(gt_df, test_df)

        # Create an aggregator of metrics (összesítő)
        mh = mm.metrics.create()
        # compute aggr of metrics
        summary = mh.compute(
            acc,
            metrics=mm.metrics.motchallenge_metrics,
            name='acc'
        )
        
        # rendering results in readable format
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        
        print(strsummary)
        
        # precision, recall values
        precision = summary.loc['acc', 'precision']
        recall = summary.loc['acc', 'recall']
        # if both > 0 --> F1 Score-t
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[tracking_csv.name] = {
            'MOTA': summary.loc['acc', 'mota'],
            'MOTP': summary.loc['acc', 'motp'],
            'Precision': precision,
            'Recall': recall,
            'F1': f1_score
        }
        
    return results


# Testing
ground_truth_file = Path('/notebooks/ObjectDetectionTracking_PN/datas/ground_truth/gt.txt')
trackings_csv_path = Path('/notebooks/ObjectDetectionTracking_PN/datas/trackings')

all_results = eval_results_for_all_trackers(trackings_csv_path, ground_truth_file)
print("All computed metrics:")
for tracker_name, metrics in all_results.items():
    print(tracker_name)
    print(metrics)