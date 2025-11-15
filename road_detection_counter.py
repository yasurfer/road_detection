import cv2 as cv
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Load YOLOv8 models
car_model = YOLO('models/yolov8n.pt')  # For car detection

# Open video file
video_path = "videos/test_dashcam.mp4"
cap = cv.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

# Setup video writer
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('videos/road_detection_result.avi', fourcc, fps, (width, height))

# Global variables
road_mask = None
counting_zone = None
road_detected = False

# Car counting variables
car_count = 0
car_tracker = defaultdict(dict)
frame_count = 0
next_car_id = 0

# Pothole counting variables
pothole_count = 0

def detect_potholes_edge(frame):
    """Pothole detection using edge detection and contour analysis"""
    # Define ROI - focus on road ahead, not the car itself
    height, width = frame.shape[:2]
    roi_y_start = height // 6  # Start from 1/6 down (higher up)
    roi_height = height // 2  # Cover middle half (road ahead, not car)
    roi_frame = frame[roi_y_start:roi_y_start + roi_height, :]
    
    # Convert to grayscale
    gray = cv.cvtColor(roi_frame, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection to find edges
    edges = cv.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    pothole_detections = []
    
    for contour in contours:
        # Filter contours by area (potholes are usually small)
        area = cv.contourArea(contour)
        if 100 < area < 5000:  # Adjust these values based on pothole size
            # Get bounding rectangle
            x, y, w, h = cv.boundingRect(contour)
            
            # Adjust coordinates back to original frame
            y += roi_y_start
            
            # Check if it's roughly circular/oval (potholes are usually irregular circles)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:  # Not too elongated
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Filter for road area only
                if is_point_in_road(center_x, center_y):
                    pothole_detections.append(((x, y, w, h), 0.8))  # High confidence
    
    return pothole_detections

def is_point_in_road(x, y):
    """Check if a point is inside the road trapezoid"""
    global road_mask
    if road_mask is not None:
        height, width = road_mask.shape
        if 0 <= x < width and 0 <= y < height:
            return road_mask[y, x] == 255
    return False

def detect_road_trapezoid(frame):
    """Detect road using trapezoid (triangle-like) shape"""
    global road_mask, counting_zone, road_detected
    
    height, width = frame.shape[:2]
    
    # Create trapezoidal road mask (very narrow at top)
    road_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define trapezoid points for road - VERY NARROW at top
    # Top edge (extremely narrow) - where road meets horizon
    top_left = (width // 2 - 20, height // 3)  # Lower position (was height // 4)
    top_right = (width // 2 + 20, height // 3)
    
    # Bottom edge (wider) - at bottom of frame (near car)
    bottom_left = (150, height - 50)
    bottom_right = (width - 150, height - 50)
    
    # Create trapezoidal road mask
    trapezoid_points = np.array([
        [top_left[0], top_left[1]],
        [top_right[0], top_right[1]], 
        [bottom_right[0], bottom_right[1]],
        [bottom_left[0], bottom_left[1]]
    ], np.int32)
    
    cv.fillPoly(road_mask, [trapezoid_points], 255)
    
    # Apply some morphological operations to clean up
    kernel = np.ones((15, 15), np.uint8)
    road_mask = cv.morphologyEx(road_mask, cv.MORPH_CLOSE, kernel)
    road_mask = cv.morphologyEx(road_mask, cv.MORPH_OPEN, kernel)
    
    # Create counting zone in the middle of the detected road
    if road_mask is not None:
        # Find the bounding box of the road area
        road_coords = np.where(road_mask == 255)
        if len(road_coords[0]) > 0:
            y_min, y_max = np.min(road_coords[0]), np.max(road_coords[0])
            x_min, x_max = np.min(road_coords[1]), np.max(road_coords[1])
            
            # Create counting zone in the middle third of the road
            zone_height = max(60, (y_max - y_min) // 4)
            zone_y = y_min + (y_max - y_min) // 3
            zone_x = x_min
            zone_width = x_max - x_min
            
            counting_zone = {'x': zone_x, 'y': zone_y, 'width': zone_width, 'height': zone_height}
            road_detected = True
            return True
    
    return False

def track_cars(current_cars, frame_count):
    """Simple car tracking based on position matching"""
    global car_count, next_car_id
    
    tracked_cars = {}
    used_current_ids = set()
    
    # Match existing tracked cars with current detections
    for car_id, car_data in car_tracker.items():
        if frame_count - car_data['last_seen'] > 10:  # Remove if not seen for 10 frames
            continue
            
        best_match = None
        min_distance = float('inf')
        
        for i, current_car in enumerate(current_cars):
            if i in used_current_ids:
                continue
                
            # Calculate center points
            current_center = ((current_car['box'][0] + current_car['box'][2]) // 2,
                            (current_car['box'][1] + current_car['box'][3]) // 2)
            tracked_center = car_data['center']
            
            # Calculate distance
            distance = np.sqrt((current_center[0] - tracked_center[0])**2 + 
                             (current_center[1] - tracked_center[1])**2)
            
            if distance < min_distance and distance < 100:  # Threshold for matching
                min_distance = distance
                best_match = i
        
        if best_match is not None:
            # Update existing car
            tracked_cars[car_id] = {
                'box': current_cars[best_match]['box'],
                'center': ((current_cars[best_match]['box'][0] + current_cars[best_match]['box'][2]) // 2,
                          (current_cars[best_match]['box'][1] + current_cars[best_match]['box'][3]) // 2),
                'last_seen': frame_count,
                'counted': car_data['counted']
            }
            used_current_ids.add(best_match)
    
    # Add new cars
    for i, current_car in enumerate(current_cars):
        if i not in used_current_ids:
            tracked_cars[next_car_id] = {
                'box': current_car['box'],
                'center': ((current_car['box'][0] + current_car['box'][2]) // 2,
                          (current_car['box'][1] + current_car['box'][3]) // 2),
                'last_seen': frame_count,
                'counted': False
            }
            next_car_id += 1
    
    return tracked_cars

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Detect road on first frame or every 30 frames
    if frame_count == 1 or frame_count % 30 == 0:
        detect_road_trapezoid(frame)
    
    # Run car detection
    car_results = car_model(frame, verbose=False)
    
    # Run pothole detection
    pothole_detections = detect_potholes_edge(frame)
    
    # Draw potholes in green
    for (x, y, w, h), confidence in pothole_detections:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Pothole {confidence:.2f}"
        cv.putText(frame, label, (x, y - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        pothole_count += 1
    
    # Extract car detections
    current_cars = []
    for result in car_results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Only detect vehicles
                if cls in [2, 5, 7, 3]:  # car, bus, truck, motorbike
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_cars.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls
                    })
    
    # Track cars
    car_tracker = track_cars(current_cars, frame_count)
    
    # Draw road mask with yellow lines
    if road_mask is not None:
        # Find contours of road area
        contours, _ = cv.findContours(road_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            # Draw road boundary in yellow
            cv.drawContours(frame, contours, -1, (0, 255, 255), 3)  # Yellow color (BGR)
            
            # Add "ROAD" label
            if contours:
                largest_contour = max(contours, key=cv.contourArea)
                if cv.contourArea(largest_contour) > 1000:
                    x, y, w, h = cv.boundingRect(largest_contour)
                    cv.putText(frame, "ROAD DETECTED", (x + 10, y + 30), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw detections and tracking info
    for car_id, car_data in car_tracker.items():
        x1, y1, x2, y2 = car_data['box']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if is_point_in_road(center_x, center_y):
            color = (0, 0, 255)  # Red for cars
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw just "Car" label (no ID number)
            cv.putText(frame, "Car", (x1, y1 - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw status info
    status_text = f"Road: {'Detected' if road_detected else 'Detecting...'}"
    cv.putText(frame, status_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv.putText(frame, f"Cars: {car_count}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame, f"Potholes: {pothole_count}", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Write frame
    out.write(frame)
    
    # Display frame
    cv.imshow('Road Detection with Car Counting', frame)
    
    # Break on 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv.destroyAllWindows()

print(f"Processing complete!")
print(f"Total cars counted: {car_count}")
print(f"Total potholes detected: {pothole_count}")
