import cv2
import random
from ultralytics import YOLO

yolo = YOLO("yolov8s.pt")

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "snowStatue.mp4"
videoCap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter.fourcc(*'MPEG')  # Codec
out = cv2.VideoWriter('output.mp4', fourcc, 24, (1280, 720))

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if not ret:
        break
    results = yolo.track(frame, stream=True)

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                class_name = class_names[cls]

                conf = float(box.conf[0])

                colour = getColours(cls)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
    out.write(frame)

videoCap.release()
out.release()
cv2.destroyAllWindows()