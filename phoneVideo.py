# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 21:55:06 2023

@author: chloe
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:49:42 2023

@author: sodapop
"""

# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
from PIL import Image, ImageOps
import tensorflow as tf





# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.1.4:8080/shot.jpg"

model = tf.keras.models.load_model('model16.h5')

def import_and_predict(image_data, model):
    
        size = (75,75)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        
        img_reshape = image[np.newaxis,...]
        
        preds = model.predict(img_reshape)
        # create a list containing the class labels
        class_labels = ['lemon','tomato','unknown']
        # find the index of the class with maximum score
        pred = np.argmax(preds, axis=-1)
        
        
        return class_labels[pred[0]]

# While loop to continuously fetching data from the Url
while True:
    
    
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    original = cv2.imdecode(img_arr, -1)
    original = imutils.resize(original, width=1000, height=1000)
    
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')
    
    #call prediction function
    prediction = import_and_predict(image, model)
    
    if prediction == 'lemon':
        predict="It is a lemon"
    elif prediction == 'tomato':
        predict="It is a tomato"
    else:
        predict="It is unknown"
        
    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Android_cam", original)

    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()