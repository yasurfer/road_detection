# Road Detection & Counter System

A Python-based computer vision system that detects roads, counts vehicles, and identifies potholes in dashcam footage using YOLOv8 and OpenCV.

## Features

- **Road Detection**: Trapezoidal road area detection with yellow visualization
- **Vehicle Counting**: Detects and tracks cars, buses, trucks, and motorcycles
- **Pothole Detection**: Edge detection-based pothole identification
- **Smart Filtering**: Only shows detections within the road area
- **Real-time Processing**: Efficient frame-by-frame analysis

## Project Structure

```
pothole-detection-main/
├── road_detection_counter.py  # Main detection script
├── models/
│   └── yolov8n.pt           # YOLOv8 model for vehicle detection
├── videos/
│   ├── test_dashcam.mp4     # Input video
│   └── road_detection_result.avi  # Output video
├── requirements.txt         # Python dependencies
└── venv/                    # Virtual environment
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the road detection system:
```bash
python road_detection_counter.py
```

The system will:
- Process the dashcam video
- Display real-time detection with road area in yellow
- Show vehicles in red and potholes in green
- Save results to `videos/road_detection_result.avi`

## Detection Details

### Road Detection
- Uses trapezoidal mask (narrow at top, wide at bottom)
- Matches road perspective from dashcam view
- Filters all detections to road area only

### Vehicle Detection
- Detects: cars, buses, trucks, motorcycles
- Tracks vehicles across frames
- Displays in red rectangles
- Simple "Car" labels (no tracking IDs)

### Pothole Detection
- Edge detection and contour analysis
- Circular/oval shape filtering
- Displays in green rectangles
- Only detects within road area

## Controls

- Press 'q' to quit the application
- Close window to stop processing

## Output

- Console: Shows total cars and potholes detected
- Video: Saves processed video with all detections
- Display: Real-time visualization with road area and detections



