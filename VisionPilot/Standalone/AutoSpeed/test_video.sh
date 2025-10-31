#!/bin/bash
# Quick test script for AutoSpeed standalone inference

# Suppress GStreamer warnings
export GST_DEBUG=1

VIDEO_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/Road_Driving_Scenes_Normal.mp4"
MODEL_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/models/AutoSpeed_n.onnx"
PRECISION="fp16"
REALTIME="true"           # Real-time playback (matches video FPS)
MEASURE_LATENCY="true"    # Enable latency metrics

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Starting AutoSpeed standalone inference..."
echo "Video: $VIDEO_PATH"
echo "Model: $MODEL_PATH"
echo "Precision: $PRECISION"
echo ""
echo "Press 'q' in the video window to quit"
echo "=========================================="
echo ""

./build/autospeed_infer_stream "$VIDEO_PATH" "$MODEL_PATH" "$PRECISION" "$REALTIME" "$MEASURE_LATENCY"

