import cv2

img = cv2.imread('../images/image.jpg')

# resize the image to 20% of original size to fit on the screen
img = cv2.resize(img, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

cv2.imshow("image", img)
cv2.waitKey(0)