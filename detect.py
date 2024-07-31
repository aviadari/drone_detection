from ultralytics import YOLOv10
# import torch

# model = YOLOv10("best.pt")
# # results2 = model(source='1.mp4', conf=0.5, stream=True, show=True)
# results = model.track(source="1.mp4", conf=0.3, iou=0.5, stream=False, save=True, show=True)
# a=1
# torch.cuda.set_device(0)

### orig ###
# model = YOLOv10("ir_s_5.pt")
# model.to('cuda')
# # results2 = model(source='1.mp4', conf=0.5, stream=True, show=True)
# results = model.track(source="SBS_People_left.avi", conf=0.1, iou=0.5, stream=False, save=True, show=True)
# a=1
##############

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLOv10

# Load the YOLOv8 model
model = YOLOv10("ir_s_5.pt")
model.to('cuda')

# Open the video file
video_path = "SBS_People_left.avi"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), cap.get(cv2.CAP_PROP_FPS), size)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # results = model.track(frame, imgsz=img_size, persist=True, conf=conf_level, tracker=tracker_option)
        # print(results)
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy().astype(int)
        # # Get the boxes and track IDs
        # boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()


            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 60:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=1)

            # Display the annotated frame
            cv2.imshow("YOLOv10 Tracking", annotated_frame)
            out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
