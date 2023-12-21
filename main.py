from ultralytics import YOLO
import cv2
import cvzone
import math
import time



cap = cv2.VideoCapture("C:\\Users\\BAB AL SAFA\\Desktop\\MINE\\LicensePlateDetection\\videos\\traffic_1.mp4")  # For Video

model = YOLO("C:\\Users\\BAB AL SAFA\\Desktop\\MINE\\LicensePlateDetection\\weights\\best.pt")

#Adding car to the detection
model_obj = YOLO("C:\\Users\\BAB AL SAFA\\Desktop\\MINE\\LicensePlateDetection\\weights\\yolov8n.pt")

classNames = ['0', 'object']
class_default = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird','cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kites', 'baseball bat',
              'baseball glove', 'skateboard', 'surf board', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwitch', 'orange', 'broccoli',
              'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
              'dinningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    img = cv2.resize(img, (1000, 600))
    results = model(img, stream=True)
    person = model_obj(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h),colorR=(0,255,0), colorC=(0,255,0),rt=1)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            if conf > 0.4:
                cvzone.putTextRect(img, f'plate number {conf}', (max(0, x1), max(35, y1)), scale=1,
                thickness=1, colorB=(0,255,0), colorT=(0,0,0), colorR=(0,255,0))


# car detection class
    for r in person:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255,0,0), colorC=(255, 0, 0), rt=1)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            if conf > 0.6:
                cvzone.putTextRect(img, f'{class_default[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                thickness=1, colorB=(255,0,0), colorT=(0,0,0), colorR=(255,0,0))


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)