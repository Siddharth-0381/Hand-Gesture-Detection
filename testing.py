import cv2  # for open cv library functions
from cvzone.HandTrackingModule import HandDetector  # To detect the hand in the image
import numpy as np  # For forming an image of same size
import math
import tensorflow as tf
import streamlit as st

model = tf.keras.models.load_model("hGDetection.h5")
st.title("Hand Gesture Detection")
html_table = """<p>Submitted By : </p><br>
                        <table>
                          <tr>
                                <th>Name</th>
                                <th>PRN</th>
                                <th>Email</th>
                          </tr>
                          <tr>
                                <td>Siddharth Kanikdale</td>
                                <td>0120190145</td>
                                <td>spkanikdale@mitaoe.ac.in</td>
                          </tr>
                          <tr>
                                <td>Vedant Dawange</td>
                                <td>PRN</td>
                                <td>Email</td>
                          </tr>
                          <tr>
                                <td>Purva Potdukhe</td>
                                <td>PRN</td>
                                <td>Email</td>
                          </tr>
                          <tr>
                                <td>Anushka Yadav</td>
                                <td>PRN</td>
                                <td>aayadav@mitaoe.ac.in</td>
                          </tr>
                      </table>"""
st.sidebar.markdown(html_table, unsafe_allow_html=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)
offset = 20  # To capture entire hand
imageSize = 300

label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I love You', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Yes', 'Z']

while True:
    success, img = cap.read()
    imgOutput = img
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
            imgTest = cv2.resize(imgWhite, (128, 128), interpolation=cv2.INTER_AREA)
            imgTest = cv2.cvtColor(imgTest, cv2.COLOR_RGB2GRAY)
            imgTest = tf.expand_dims(imgTest, axis=0)
            prediction = model.predict(imgTest)
            print(prediction)

        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            imgTest = cv2.resize(imgWhite, (128, 128), interpolation=cv2.INTER_AREA)
            imgTest = cv2.cvtColor(imgTest, cv2.COLOR_RGB2GRAY)
            imgTest = tf.expand_dims(imgTest, axis=0)
            prediction = model.predict(imgTest)
            print(prediction)

        max_value = np.argmax(prediction)
        cv2.putText(imgOutput, label[max_value], (x, y - 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=4,
                    color=(255, 0, 255), fontScale=1)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset),
                      (255, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
