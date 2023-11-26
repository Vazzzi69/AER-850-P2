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

# revisit the output channels sizes

## im going to try different kind of activation'
from tensorflow.keras.layers import LeakyReLU

##im going to add a batch normalization first to each layer
from tensorflow.keras.layers import BatchNormalization

#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
model.add(layers.Conv2D(16, (3, 3), input_shape=image_shape))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))

#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.Conv2D(32, (3, 3)))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.01))
#model.add(layers.MaxPooling2D((2, 2)))


#added a layer to see if performance increases

# Dense Layer & Hyper Param tuning

model.add(layers.Flatten())

model.add(layers.Dense(32, activation='relu'))

#model.add(layers.Dense(32))
model.add(BatchNormalization())
#model.add(layers.LeakyReLU(alpha=0.01)) ## maybe chage this

model.add(layers.Dropout(0.1))  #reduce the rate to see if its better?
model.add(layers.Dense(4,activation='softmax'))


model.summary()

# Make the model 

#Adding custom optimizer

adam = tf.keras.optimizers.Adam(learning_rate=0.01) # slows down the learning rate going down

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#STEP 4:  Model Evaluation



import matplotlib.pyplot as plt

# increase it 20, to see if accuracy 

history = model.fit(train_generator,epochs=20, validation_data=
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

# STEP 5: Model Testing 

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt



### LARGE ONE

test_image_path = 'Project 2 Data/Data/Test/Large/Crack__20180419_13_29_14,846.bmp'  
test_image = image.load_img(test_image_path, target_size=(100, 100))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array /= 255. 

predictions = model.predict(test_image_array)

predicted_class_index = np.argmax(predictions)

class_labels = {0: 'small', 1: 'medium', 2: 'large', 3: 'none'}
predicted_class_label = class_labels[predicted_class_index]

plt.imshow(test_image)
plt.title(f'Predicted Class classification label: {predicted_class_label}')
plt.axis('off')

plt.text(0, -10, 'True Crack classification label: Large ', color='black', fontsize=12,)


for i, prob in enumerate(predictions[0]):
    label = class_labels[i]
    plt.text(60, 80 + i * 6, f'{label}: {prob:.2%}', color='green', fontsize=10)

plt.show()

#medium one

test_image_path = 'Project 2 Data/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp'  
test_image = image.load_img(test_image_path, target_size=(100, 100))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array /= 255.  # Normalize the image by dividing by 255

predictions = model.predict(test_image_array)

predicted_class_index = np.argmax(predictions)

class_labels = {0: 'small', 1: 'medium', 2: 'large', 3: 'none'}
predicted_class_label = class_labels[predicted_class_index]

plt.imshow(test_image)
plt.title(f'Predicted Class classification label: {predicted_class_label}')
plt.axis('off')

plt.text(0, -10, 'True Crack classification label: Medium ', color='black', fontsize=12,)


for i, prob in enumerate(predictions[0]):
    label = class_labels[i]
    plt.text(60, 80 + i * 6, f'{label}: {prob:.2%}', color='green', fontsize=10)

plt.show()

