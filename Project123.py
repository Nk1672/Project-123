import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import os, ssl, time
import PIL.ImageOps

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml('mnist_784',version = 1, return_X_y = True)
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
print('Accuracy: ',accuracy_score(y_test, y_pred))

cap = cv2.VideoCapture(0)

while True:
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        height,width = gray.shape
        upperLeft = (int(width/2 - 56),int(height/2 - 56))
        bottomRight = (int(width/2 + 56),int(height/2 + 56))
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)

        roi = gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]

        img_pil = Image.fromarray(roi)
        
        img = img_pil.convert('L')
        img_resize = img.resize((22,22),Image.ANTIALIAS)
        img_resize_inverted = PIL.ImageOps.invert(img_resize)
        pixelFilter = 20
        
        minPixel = np.percentile(img_resize_inverted, pixelFilter)
        img_resize_inverted_scaled = np.clip(img_resize_inverted-minPixel, 0, 255)
        maxPixel = np.max(img_resize_inverted)
        img_resize_inverted_scaled = np.asarray(img_resize_inverted_scaled)/maxPixel 
        
        test_sample = np.array(img_resize_inverted_scaled).reshape(1, 784)
        test_predict = lr.predict(test_sample)
        print('Predicted Class Is: ',test_predict)

        #Display the result frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    except Exception as e:
        pass

#Releasing the capture
cap.release()
cap.destroyAllWindows()
        