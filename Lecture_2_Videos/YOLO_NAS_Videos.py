import math

import cv2
import torch
import numpy as np
from super_gradients.training import models

cap = cv2.VideoCapture("../videos/bikes.mp4")


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = models.get('yolo_nas_s', pretrained_weights='coco').to(device)


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
while cap.isOpened():
    ret_, frame = cap.read()
    count+=1
    if count % 20 == 0:
        if ret_:
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
                label = f"{classNames[int(cls)]}{conf}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                print("Frame N ", count, "", x1, y1, x2, y2, label, c2, t_size)

                cv2.rectangle(frame, (x1,y1), c2, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF==ord('1'):
                break
#     else:
#         break
#
cap.release()
cv2.destroyAllWindows()



