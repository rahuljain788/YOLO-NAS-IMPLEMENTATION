import cv2

cap = cv2.VideoCapture("../videos/bikes.mp4")

while cap.isOpened():
    ret_, frame = cap.read()
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF==ord('1'):
        break

cap.release()
cv2.destroyAllWindows()