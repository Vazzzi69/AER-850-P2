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
    shear_range=0.2,                        #might change val?
    zoom_range=0.2,                         #might change val?
    
    horizontal_flip=True )


validation_datagen = ImageDataGenerator(rescale=1./255)   


# Training data generator

train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=image_shape[:2], #reinput size 
    batch_size=32,
    shuffle =True,    # to randomize
    class_mode='categorical'
)

# Validation data generator

validation_generator = validation_datagen.flow_from_directory(
    validation_data_directory,
    target_size=image_shape[:2],#reinput size
    batch_size=32,
    class_mode='categorical'
)

# STEP 2: Neural Network Architecture Design & STEP 3: Hyperparameter analysis

import tensorflow as tf
from tensorflow.keras import layers, models

# Creating the model layers

model = models.Sequential()

# Trying LEakyReLU function instead of ReLU

from tensorflow.keras.layers import LeakyReLU

# Adding Batch Normalization to each layer

from tensorflow.keras.layers import BatchNormalization

model.add(layers.Conv2D(16, (3, 3), input_shape=image_shape))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))

# Second layer wasnt needed either 

model.add(layers.Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))

# Third layer isnt required for this calssifcation too complex 

# Dense Layer & Hyper Param tuning

model.add(layers.Flatten())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.1))  #Reduced from 0.5 initally 
model.add(layers.Dense(4,activation='softmax'))


model.summary()

# Model Creation 

# Adding Custom optimizer to tune learning rate

adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#STEP 4:  Model Evaluation


import matplotlib.pyplot as plt

# Increased from epoch=10 to Epoch=20

history = model.fit(train_generator,epochs=25, validation_data=
                    validation_generator)

# Accuracy Plot

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Plot

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Saving Model 

model.save('finaltest.keras')