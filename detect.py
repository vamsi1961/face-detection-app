import cv2
import os
import random
import numpy as np
import uuid

from matplotlib import pyplot as plt
# Import tensorflow dependencies functional API


from tensorflow.keras.models import Model
# In Model u pass inputs and outputs "Model(inputs=[inputimage, verificatioimage] ,outputss[1,0])"
#
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,Input,Flatten

# Layer class is a high level layer allows us to define a custom layer and we can create a whole new class and generate a new Layer
# class L1Dist(Layer)

# Input says what input is to be given

# Flatten => COnvert convolutinal layer to dense layer

import tensorflow as tf
# 
# setup paths/directories
POS_PATH = os.path.join('data' , 'positive')
NEG_PATH = os.path.join('data' , 'negative')
ANC_PATH = os.path.join('data' , 'anchor')


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    # cut down frame to 250*250
    frame = frame[120:120+250,200:200+250, : ]


    # collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a') :
        # create a unique file path
        imgname = os.path.join(ANC_PATH , '{}.jpg'.format(uuid.uuid1()))
        # write out anchor image
        cv2.imwrite(imgname,frame)

    # collect positives
    if cv2.waitKey(1) & 0XFF == ord('p') :
        # create a unique file path
        imgname = os.path.join(POS_PATH , '{}.jpg'.format(uuid.uuid1()))
        # write out anchor image
        cv2.imwrite(imgname,frame)

    

    cv2.imshow("Image" , frame)

    if cv2.waitKey(1) & 0XFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()




