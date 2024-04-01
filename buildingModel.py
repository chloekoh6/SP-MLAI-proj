# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:01:19 2023

@author: chloe
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

train_directory = 'C:/MLAI/train'
valid_directory = 'C:/MLAI/valid'
test_directory = 'C:/MLAI/test'


train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",)

test_datagen = ImageDataGenerator(rescale=1 / 255.0) 
valid_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_generator = train_datagen.flow_from_directory(directory = train_directory,
                                                    target_size = (75,75),
                                                    batch_size = 40,
                                                    class_mode = 'categorical',
                                                    shuffle = True)

valid_generator = train_datagen.flow_from_directory(directory = valid_directory,
                                                  target_size = (75,75),
                                                  batch_size = 20,
                                                  class_mode = 'categorical',
                                                  shuffle = True)

test_generator = test_datagen.flow_from_directory(directory = test_directory,
                                                  target_size=(75,75),
                                                  batch_size = 20,
                                                  class_mode = 'categorical',
                                                  shuffle = False)


model = keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape = (75,75,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
  ]
)

model.compile(
    keras.optimizers.Adam(1e-4), loss = 'categorical_crossentropy', metrics=['accuracy']
)

history = model.fit(
    train_generator, validation_data=(valid_generator), epochs=10
)

model.evaluate(test_generator)
model.summary()

keras.models.save_model(model, 'model16.h5')



accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'y', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'g', label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()