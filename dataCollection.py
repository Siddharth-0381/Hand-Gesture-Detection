# for open cv library functions
import cv2
from cvzone.HandTrackingModule import HandDetector  # To detect the hand in the image
import numpy as np  # For forming an image of same size
import math
import time  # For saving image with different name

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)
offset = 20  # To capture entire hand
imageSize = 300

folder = "Data/A"
counter = 0

filter = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imageSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_RGB2GRAY)
        imgWhite = cv2.filter2D(imgWhite, ddepth=-1, kernel=filter)

        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)