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

# STEP 2: Neural Network Architecture Design 

import tensorflow as tf
from tensorflow.keras import layers, models

# Creating the model layers

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Dense Layer

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4,activation='softmax'))



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
