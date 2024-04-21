import motmetrics as mm
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

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
    
    print("\n First few rows of the loaded ground truth data:")
    print(gt_df.head())
    return gt_df

def load_tracking_data(trackings_csv_path: str) -> pd.DataFrame:
    try:
        tracking_df = pd.read_csv(trackings_csv_path)   # read csv assuming the first row is head
        tracking_df['frame_id'] = tracking_df['frame_id'].astype(int)#convert the 'frame_id' column to an integer for consistent processing
        
        print("\n First few rows of the loaded trackings data:")
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

    tracker_names = {  # dictionary to map filenames to tracker names
        'park_people_botsort.csv': 'BotSORT',
        'park_people_bytetrack.csv': 'ByteTrack',
        'park_people_deepsort.csv': 'DeepSort'
    }
    
    for tracking_csv in trackings_csv_path.glob('*.csv'):
        print(f"\n Evaluating {tracking_csv.name} \n")
        test_df = load_tracking_data(str(tracking_csv))
        acc = compute_motmetrics(gt_df, test_df)
        
        mh = mm.metrics.create() #Create an aggregator of metrics
        
        summary = mh.compute( #compute aggregate of metrics
            acc,
            metrics=mm.metrics.motchallenge_metrics,
            name='acc'
        )        
    
        strsummary = mm.io.render_summary( # rendering results in readable format
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        
        print(strsummary)
        
        # Calculate precision, recall, and F1 score
        precision = summary.loc['acc', 'precision']
        recall = summary.loc['acc', 'recall']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        tracker_name = tracker_names.get(tracking_csv.name, 'Unknown Tracker')
            
        # Metrics for export
        results[tracker_names.get(tracking_csv.name, 'Unknown Tracker')] = {
            'MOTA': summary.loc['acc', 'mota'],
            'MOTP': summary.loc['acc', 'motp'],
            'Precision': summary.loc['acc', 'precision'],
            'Recall': summary.loc['acc', 'recall'],
            'F1': f1_score,
            'IDF1': summary.loc['acc', 'idf1'],
            'ML': summary.loc['acc', 'num_misses'], 
            'FP': summary.loc['acc', 'num_false_positives'],
            'FN': summary.loc['acc', 'num_misses'],
            'IDs': summary.loc['acc', 'num_switches'],
            'Fragmentation': summary.loc['acc', 'num_fragmentations']
        }
    
    return results

def plot_and_save_metrics(all_results, metrics_path, name_mapping):
    display_trackers = [name_mapping.get(tracker, tracker) for tracker in all_results.keys()] # name_mapping to replace keys with the desired display names
    f1_scores = [all_results[tracker]['F1'] for tracker in all_results.keys()]
    mota_scores = [all_results[tracker]['MOTA'] for tracker in all_results.keys()]
    motp_scores = [all_results[tracker]['MOTP'] for tracker in all_results.keys()]
    recall_scores = [all_results[tracker]['Recall'] for tracker in all_results.keys()]
    precision_scores = [all_results[tracker]['Precision'] for tracker in all_results.keys()]
    idf1_scores = [all_results[tracker]['IDF1'] for tracker in all_results.keys()]
    ml_scores = [all_results[tracker]['ML'] for tracker in all_results.keys()]
    fp_scores = [all_results[tracker]['FP'] for tracker in all_results.keys()]
    fn_scores = [all_results[tracker]['FN'] for tracker in all_results.keys()]
    ids_scores = [all_results[tracker]['IDs'] for tracker in all_results.keys()]
    fragmentation_scores = [all_results[tracker]['Fragmentation'] for tracker in all_results.keys()]

    
    # F1 Score diagram
    plt.figure(figsize=(10, 5))
    plt.bar(display_trackers, idf1_scores, color='#89cff0')
    plt.xlabel('\n Objektumkövető algoritmusok')
    plt.ylabel('F1 pontszám')
    plt.title('Az objektumkövetők F1 pontszámai  \n')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(metrics_path + 'F1_Scores.png')  # Save as PNG 
    plt.close()
    
     # IDF1 Score diagram
    plt.figure(figsize=(10, 5))
    plt.bar(display_trackers, f1_scores, color='#b6ff87')
    plt.xlabel('\n Objektumkövető algoritmusok')
    plt.ylabel('IDF1 pontszám')
    plt.title('Az objektumkövetők IDF1 pontszámai  \n')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(metrics_path + 'IDF1_Scores.png')  # Save as PNG 
    plt.close()

    # MOTA diagram
    plt.figure(figsize=(10, 5))
    plt.bar(display_trackers, mota_scores, color='#FFE697')
    plt.xlabel('\n Objektumkövető algoritmusok')
    plt.ylabel('MOTA pontszám')
    plt.title('Az objektumkövetők MOTA pontszámai  \n')
    plt.ylim([min(mota_scores) - 0.1, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(metrics_path + 'MOTA_Scores.png') 
    plt.close()
    
    # Prepare DataFrame for Excel export
    metrics_data = {
        'Tracker': display_trackers,
        'MOTA': mota_scores,
        'MOTP': motp_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1': f1_scores,
        'IDF1': idf1_scores,
        'Mostly Lost (ML)': ml_scores,
        'False Positives (FP)': fp_scores,
        'False Negatives (FN)': fn_scores,
        'ID Switches (IDs)': ids_scores,
        'Fragmentation': fragmentation_scores
    }
    
    df = pd.DataFrame(metrics_data)
    df.set_index('Tracker', inplace=True)

    # DataFrame mentése Excelbe
    excel_path = f'{metrics_path}/Tracking_Results.xlsx'
    df.to_excel(excel_path)

    print(f'Results saved to {metrics_path}')

name_mapping = {
    'park_people_botsort.csv': 'BotSORT',
    'park_people_bytetrack.csv': 'ByteTrack',
    'park_people_deepsort.csv': 'DeepSort'
}

# Testing
metrics_path = '/notebooks/ObjectDetectionTracking_PN/datas/metrics/'
ground_truth_file = Path('/notebooks/ObjectDetectionTracking_PN/datas/ground_truth/gt.txt')
trackings_csv_path = Path('/notebooks/ObjectDetectionTracking_PN/datas/trackings')

# calculating metrics, diagrams
all_results = eval_results_for_all_trackers(trackings_csv_path, ground_truth_file)
plot_and_save_metrics(all_results, metrics_path, name_mapping)

print("All computed metrics:")
for tracker_name, metrics in all_results.items():
    print(tracker_name)
    print(metrics)