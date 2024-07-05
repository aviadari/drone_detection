from ultralytics import YOLOv10

model = YOLOv10("best.pt")
# results2 = model(source='1.mp4', conf=0.5, stream=True, show=True)
results = model.track(source="1.mp4", conf=0.3, iou=0.5, stream=False, save=True, show=True)
