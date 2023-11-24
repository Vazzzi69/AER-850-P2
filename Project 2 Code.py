# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:54:14 2023

@author: Vasi
"""

#STEP 1: Data Processing

from keras.preprocessing.image import ImageDataGenerator

# Image shape definition

image_shape = (100, 100, 3)

# Establishing Directory 

train_data_directory = 'Project 2 Data\Data\Train'
validation_data_directory = 'Project 2 Data\Data\Validation'
test_data_directory = 'Project 2 Data\Data\Test'

# Data Augmentation

train_datagen = ImageDataGenerator(
    
    rescale=1./255,                         #might change val?
    shear_range=0.1,                        #might change val?
    zoom_range=0.1,                         #might change val?
    
    horizontal_flip=True )


validation_datagen = ImageDataGenerator(rescale=1./255)   #might change val?


# Training data generator

train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=image_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

# Validation data generator

validation_generator = validation_datagen.flow_from_directory(
    validation_data_directory,
    target_size=image_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

# STEP 2: Neural Network Architecture Design & STEP 3: Hyperparameter analysis

import tensorflow as tf
from tensorflow.keras import layers, models

# Creating the model layers

model = models.Sequential()

# revisit the output channels sizes

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #check number
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # check number 


# Dense Layer & Hyper Param tuning

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4,activation='softmax'))


model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#STEP 4:  Model Evaluation



import matplotlib.pyplot as plt


history = model.fit(train_generator,epochs=10, validation_data=
                    validation_generator)



plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()




