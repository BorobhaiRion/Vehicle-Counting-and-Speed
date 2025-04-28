from ultralytics import YOLO
import cv2
import pandas as pd
import time

# Step 1: Load the YOLOv8 Nano model
model = YOLO('yolov8n.pt')  # Using YOLOv8n for now

# Step 2: Open the MP4 video
cap = cv2.VideoCapture('tf.mp4')  # Replace 'tf.mp4' with your actual file name

# Step 3: Prepare to save vehicle data
vehicle_data = []
up_count = 0
down_count = 0

# Step 4: Set lines for up and down directions
up_line_y = 150
down_line_y = 300

# Dictionary to store previous positions for each vehicle
previous_positions = {}

# Real distance between lines (in meters) - adjust this to your real-world setting
real_distance_meters = 10

# Frame skipping variable (skip 2 frames to speed up)
frame_skip = 2
frame_count = 0

# Record the start time of video processing
video_start_time = time.time()

# Step 5: Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 360))

    results = model.track(frame, persist=True, classes=[2, 5, 7])  # 2=car, 5=bus, 7=truck

    cv2.line(frame, (0, up_line_y), (frame.shape[1], up_line_y), (255, 0, 0), 2)
    cv2.line(frame, (0, down_line_y), (frame.shape[1], down_line_y), (0, 255, 0), 2)

    if results[0].boxes.id is not None:
        for box, track_id, cls in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            track_id = int(track_id)
            vehicle_type = {2: "Car", 5: "Bus", 7: "Truck"}.get(int(cls), "Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{track_id} ({vehicle_type})", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            if track_id not in previous_positions:
                previous_positions[track_id] = {'y': cy, 'time': time.time(), 'counted': False}

            prev_y = previous_positions[track_id]['y']
            prev_time = previous_positions[track_id]['time']

            current_time = time.time()
            elapsed_time = current_time - video_start_time  # New: relative timestamp from start

            if not previous_positions[track_id]['counted'] and prev_y > up_line_y and cy <= up_line_y:
                time_taken = current_time - prev_time
                speed = (real_distance_meters / time_taken) * 3.6

                vehicle_data.append([track_id, 'Up', vehicle_type, round(speed, 2), round(elapsed_time, 2)])
                previous_positions[track_id]['counted'] = True
                up_count += 1

                cv2.putText(frame, f"Speed: {round(speed, 2)} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if not previous_positions[track_id]['counted'] and prev_y < down_line_y and cy >= down_line_y:
                time_taken = current_time - prev_time
                speed = (real_distance_meters / time_taken) * 3.6

                vehicle_data.append([track_id, 'Down', vehicle_type, round(speed, 2), round(elapsed_time, 2)])
                previous_positions[track_id]['counted'] = True
                down_count += 1

                cv2.putText(frame, f"Speed: {round(speed, 2)} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            previous_positions[track_id]['y'] = cy
            previous_positions[track_id]['time'] = current_time

    cv2.putText(frame, f"Up: {up_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Down: {down_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Step 6: Save the results

cap.release()
cv2.destroyAllWindows()

if len(vehicle_data) > 0:
    df = pd.DataFrame(vehicle_data, columns=['Track_ID', 'Direction', 'Vehicle_Type', 'Speed_kmph', 'Timestamp'])

    # Sort so that all 'Up' vehicles first, then 'Down', both sorted by timestamp
    df['Direction'] = pd.Categorical(df['Direction'], categories=['Up', 'Down'], ordered=True)
    df = df.sort_values(by=['Direction', 'Timestamp']).reset_index(drop=True)

    df.to_csv('vehicle_output.csv', index=False)
    print("✅ Done! Results saved to 'vehicle_output.csv'")
else:
    print("No vehicle data to save.")