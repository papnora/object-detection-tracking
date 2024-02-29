import torch
print(torch.__version__)
print(torch.cuda.is_available())

from object_detection import ObjectDetection

def main():
    weights_path = '/notebooks/ObjectDetectionTracking_PN/weights/yolov7-tiny.pt'
    video_path = '/notebooks/ObjectDetectionTracking_PN/datas/videos/hongkong_pedestrians.mp4'
    output_path = '/notebooks/ObjectDetectionTracking_PN/output/processed_video.avi' 
    detector = ObjectDetection(weights_path, video_path, output_path)
    detector.process_video()

if __name__ == '__main__':
    main()