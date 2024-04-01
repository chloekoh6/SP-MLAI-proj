# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:17:09 2023

@author: chloe
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model('model16.h5')

img = load_img('C:/MLAI/tomato.jpg', target_size=(75, 75))
x = img_to_array(img)/255

preds = model.predict(x.reshape(1,75,75,3))
# create a list containing the class labels
class_labels = ['lemon','tomato','unknown']
# find the index of the class with maximum score
pred = np.argmax(preds, axis=-1)
# print the label of the class with maximum score
imgplot = plt.imshow(img)
print(class_labels[pred[0]])

    