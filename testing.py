import cv2  # for open cv library functions
from cvzone.HandTrackingModule import HandDetector  # To detect the hand in the image
import numpy as np  # For forming an image of same size
import math
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

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

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# detector = HandDetector(maxHands=1)

class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            mp_holistic = mp.solutions.holistic # Holistic model
            mp_drawing = mp.solutions.drawing_utils # Drawing utilities

            # Actions that we try to detect
            actions = np.array(['hello', 'thanks', 'iloveyou'])

            # Load the model from Modelo folder:

            model = load_model('model.h5',actions)

            # 1. New detection variables
            sequence = []
            sentence = []
            threshold = 0.8

            # Set mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while True:
                    #img = frame.to_ndarray(format="bgr24")
                    flip_img = cv2.flip(img,1)

                    # Make detections
                    image, results = mediapipe_detection(flip_img, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        #print(actions[np.argmax(res)])

                    #3. Viz logic
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]

                        # Viz probabilities
                        image = prob_viz(res, actions, image)

                    cv2.rectangle(image, (0,0), (640, 40), (234, 234, 77), 1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    return av.VideoFrame.from_ndarray(image,format="bgr24")
      

offset = 20  # To capture entire hand
imageSize = 300

label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I love You', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Yes', 'Z']

while True:
    cap = webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )
    img = cap
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
            imgTest = st.image()
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
