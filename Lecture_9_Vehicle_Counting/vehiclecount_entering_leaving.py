import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import cv2
import torch
from super_gradients.training import models
from sort import *

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture("../videos/bikes.mp4")


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = models.get('yolo_nas_s', pretrained_weights='coco').to(device)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 +=offset[0]
        x2 +=offset[0]
        y1 +=offset[0]
        y2 +=offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        label = str(id) + ":" + classNames[cat]
        (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.rectangle(img, (x1,y1-20), (x1+w, y1), (255, 144,30), -1)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,255], 1)
    return img


classNames = ["person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
            "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
            "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

count = 0
total_count_up = 0
total_count_down = 0

while cap.isOpened():
    ret_, frame = cap.read()
    count+=1
    if count % 5 == 0:
        if ret_:
            detections = np.empty((0,5))
            # resize the image to 50% of original size to fit on the screen
            # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
            result = list(model.predict(frame, conf=0.4))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()

            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                conf = math.ceil((confidence*100)) / 100
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
            tracker_dets = tracker.update(detections)
            if len(tracker_dets) > 0:
                bbox_xyxy = tracker_dets[:, :4]
                identities = tracker_dets[:,8]
                categories = tracker_dets[:,4]
                draw_boxes(frame, bbox_xyxy, identities, categories)
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF==ord('1'):
                break
#     else:
#         break
#
cap.release()
cv2.destroyAllWindows()



