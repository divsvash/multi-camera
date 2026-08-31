from ultralytics import YOLO

model = YOLO("plate_detector_v1.pt")

results = model("test_frames/frame5.jpg")

results[0].show()          # opens a window showing the image with boxes drawn
results[0].save("output5.jpg")   # also saves it to disk so you can look at it anytime

print(results[0].boxes)    # prints raw detection data (coordinates, confidence, class)