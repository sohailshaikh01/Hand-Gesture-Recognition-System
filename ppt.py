import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 
import pyautogui as p
import time 
import wmi
import psutil

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            #//
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])

import pickle
# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0    
while True:
 
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    # print(data.shape)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    print(y_pred)
    
    # Initializing the wmi constructor
    #f = wmi.WMI()
  
    # Iterating through all the running processes
    #ch="chrome.exe" in (i.name() for i in psutil.process_iter())
    #if ch=="True":
    if y_pred=='1':
        p.press('PgDn')
        time.sleep(0.1)
    elif y_pred=='2':
        p.press('PgUp')
        time.sleep(0.1)
    elif y_pred=='3':
        p.press('Esc')
        time.sleep(0.1)
    #elif y_pred=='4':
     #   p.press('')
      #  time.sleep(0.1)
    #elif y_pred=='5':
     #   time.sleep(0.01)
    elif y_pred=='OK':
        p.press('F5')
        time.sleep(0.1)
    #elif y_pred=='PTRN8':
     #   p.press('f')
      #  time.sleep(0.1)'''
    #elif y_pred=='SS':
     #   p.press('screen_s')
      #  time.sleep(0.1)
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 5
    
    # Using cv2.putText() method
    #frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                    #fontScale, color, thickness, cv2.LINE_AA)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()