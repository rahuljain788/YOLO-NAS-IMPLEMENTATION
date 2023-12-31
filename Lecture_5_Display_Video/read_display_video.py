import cv2

cap = cv2.VideoCapture("../videos/bikes.mp4")

while cap.isOpened():
    ret_, frame = cap.read()
    if ret_:
        # resize the image to 50% of original size to fit on the screen
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()