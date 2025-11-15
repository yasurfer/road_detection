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

3. Download required files:

### Test Video
Download dashcam footage from YouTube:
```bash
# Option 1: Use yt-dlp (recommended)
pip install yt-dlp
yt-dlp "https://www.youtube.com/watch?v=BQo87tGRM74" -o videos/test_dashcam.mp4

# Option 2: Download manually from browser and save to videos/test_dashcam.mp4
```

### YOLOv8 Model
Download the YOLOv8 nano model:
```bash
# The model will be downloaded automatically when you first run the script
# Or download manually from: https://github.com/ultralytics/ultralytics/releases
# Save as: models/yolov8n.pt
```

## 📁 Files Needed

Before running the script, make sure you have:

### Required Files
- **`models/yolov8n.pt`** - YOLOv8 model (auto-downloaded on first run)
- **`videos/test_dashcam.mp4`** - Test video (download from YouTube below)

### Download Test Video
```bash
# From YouTube (dashcam footage)
pip install yt-dlp
yt-dlp "https://www.youtube.com/watch?v=BQo87tGRM74" -o videos/test_dashcam.mp4
```

### Download Model
```bash
# YOLOv8 nano model (auto-downloaded)
# Manual download: https://github.com/ultralytics/ultralytics/releases
# Save as: models/yolov8n.pt
```

### Use Your Own Files
- Place your video in `videos/your_video.mp4`
- Update the `video_path` variable in `road_detection_counter.py`

## 🚀 Getting Started

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/yasurfer/road_detection.git
cd road_detection

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download test video (optional - use your own video if preferred)
pip install yt-dlp
yt-dlp "https://www.youtube.com/watch?v=BQo87tGRM74" -o videos/test_dashcam.mp4

# 5. Run the system
python road_detection_counter.py
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



