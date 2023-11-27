# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:47:17 2023

@author: Vasi
"""

# STEP 5: Model Testing 

from tensorflow.keras.models import load_model

# Load the Saved Model 

model_path = 'C:/Users/Vasi/Documents/GitHub/AER-850-P2/finaltest.keras'
model = load_model(model_path)

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


# Large Image Classification

test_image_path = 'Project 2 Data/Data/Test/Large/Crack__20180419_13_29_14,846.bmp'  
test_image = image.load_img(test_image_path, target_size=(100, 100))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array /= 255. 

predictions = model.predict(test_image_array)

predicted_class_index = np.argmax(predictions)

class_labels = {0: 'Small', 1: 'Medium', 2: 'Large', 3: 'None'}
predicted_class_label = class_labels[predicted_class_index]

plt.imshow(test_image)
plt.title(f'Predicted Class classification label: {predicted_class_label}')
plt.axis('off')

plt.text(0, -10, 'True Crack classification label: Large ', color='black', fontsize=12,)

# Adding Percentages on image

for i, prob in enumerate(predictions[0]):
    label = class_labels[i]
    plt.text(60, 80 + i * 6, f'{label}: {prob:.2%}', color='green', fontsize=10)

plt.show()

# Medium Image Classification

test_image_path = 'Project 2 Data/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp'  
test_image = image.load_img(test_image_path, target_size=(100, 100))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array /= 255.  

predictions = model.predict(test_image_array)

predicted_class_index = np.argmax(predictions)

class_labels = {0: 'Small', 1: 'Medium', 2: 'Large', 3: 'None'}
predicted_class_label = class_labels[predicted_class_index]

plt.imshow(test_image)
plt.title(f'Predicted Class classification label: {predicted_class_label}')
plt.axis('off')

plt.text(0, -10, 'True Crack classification label: Medium ', color='black', fontsize=12,)

# Adding Percentages on image

for i, prob in enumerate(predictions[0]):
    label = class_labels[i]
    plt.text(60, 80 + i * 6, f'{label}: {prob:.2%}', color='green', fontsize=10)

plt.show()

