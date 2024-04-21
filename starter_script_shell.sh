#!/bin/bash
export PYTHONPATH=/notebooks/ObjectDetectionTracking_PN/ByteTrack/
pip install loguru lap thop cython_bbox yacs faiss-gpu motmetrics memory_profiler
python /notebooks/ObjectDetectionTracking_PN/scripts/starter_script.py
