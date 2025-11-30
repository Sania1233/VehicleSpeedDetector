import cv2
import numpy as np
import time

VIDEO_FILE = r"C:\Users\sania\OneDrive\Documents\python\vechile_speed\traffic.mp4"
DISTANCE = 12  # meters between the two lines

# Lines for speed measurement
lineA = 200
lineB = 300

# Video capture
cap = cv2.VideoCapture(VIDEO_FILE)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)

# Vehicle tracker
next_vehicle_id = 0
vehicles = {}

MAX_DIST = 50
MIN_CONTOUR_AREA = 1000
MAX_MISSING_FRAMES = 5

def distance(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

# Simple chatbot function: generates a sentence based on speed
def chatbot_response(speed):
    if speed is None:
        return "Calculating..."
    elif speed < 30:
        return f"Vehicle moving slowly at {speed:.1f} km/h"
    elif speed < 60:
        return f"Vehicle speed is moderate: {speed:.1f} km/h"
    else:
        return f"Vehicle is fast! {speed:.1f} km/h"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Draw reference lines
    cv2.line(frame, (0, lineA), (width, lineA), (255, 0, 0), 2)
    cv2.line(frame, (0, lineB), (width, lineB), (0, 255, 0), 2)

    # Background subtraction
    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w/2)
        cy = int(y + h/2)
        detections.append((cx, cy, x, y, w, h))

    used_ids = []

    # Match detections
    for det in detections:
        cx, cy, x, y, w, h = det
        matched = False
        for vid, data in vehicles.items():
            if vid in used_ids:
                continue
            if distance((cx, cy), data['centroid']) < MAX_DIST:
                prev_y = data['centroid'][1]
                vehicles[vid]['centroid'] = (cx, cy)
                vehicles[vid]['bbox'] = (x, y, w, h)
                vehicles[vid]['frames'] = 0

                # Line crossing
                if data['cross_A'] is None and ((prev_y < lineA <= cy) or (prev_y > lineA >= cy)):
                    vehicles[vid]['cross_A'] = time.time()
                if data['cross_A'] is not None and data['cross_B'] is None and ((prev_y < lineB <= cy) or (prev_y > lineB >= cy)):
                    vehicles[vid]['cross_B'] = time.time()
                    time_taken = vehicles[vid]['cross_B'] - vehicles[vid]['cross_A']
                    vehicles[vid]['speed'] = (DISTANCE / time_taken) * 3.6

                matched = True
                used_ids.append(vid)
                break

        if not matched:
            vehicles[next_vehicle_id] = {
                'centroid': (cx, cy),
                'cross_A': None,
                'cross_B': None,
                'speed': None,
                'bbox': (x, y, w, h),
                'frames': 0
            }
            next_vehicle_id += 1

    # Remove missing vehicles
    remove_ids = []
    for vid, data in vehicles.items():
        if vid not in used_ids:
            vehicles[vid]['frames'] += 1
        if vehicles[vid]['frames'] > MAX_MISSING_FRAMES:
            remove_ids.append(vid)
    for rid in remove_ids:
        vehicles.pop(rid)

    # Draw vehicles and speed
    for vid, data in vehicles.items():
        x, y, w, h = data['bbox']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        # Display speed above vehicle
        response = chatbot_response(data.get('speed'))
        cv2.putText(frame, response, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Draw detector box
    box_height = 50 + 25*len(vehicles)
    cv2.rectangle(frame, (10, 10), (400, box_height), (50,50,50), -1)
    cv2.putText(frame, "Speed Detector Chatbot", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    y0 = 60
    for vid, data in vehicles.items():
        response = chatbot_response(data.get('speed'))
        cv2.putText(frame, f"Vehicle {vid}: {response}", (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y0 += 25

    cv2.imshow("Vehicle Speed Detection Chatbot", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
