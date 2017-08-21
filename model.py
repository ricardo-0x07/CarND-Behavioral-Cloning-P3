import itertools
import csv
import cv2
import numpy as np
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dropout, Activation, Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
lines = []
images = []
measurements = []

def resize_function(image):
    """Crop and Resize image.
    Args: image: image data
    Return: cropped and resized image 
    """
    image = image[60:140]
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    return  image

# Open and read csv file
with open('./data/set3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    low_steer = 0
    steerings = []
    for line in reader:
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.23 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        # read in images from center, left and right cameras
        for i in range(3):
            source_path = line[i]
            file_name = source_path.split('/')[-1]
            current_path = './data/set3/IMG/' + file_name
            image = cv2.imread(current_path)
            if image is not None:
                image = resize_function(image)
                images.append(image)
            else:
                print('current path', current_path)
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)


augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    # Flip images
    augmented_images.append(cv2.flip(image, 1))
    # Flip steering measurement
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
# Normalize image data
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(64,64,3)))
# Add convolution layers
model.add(Conv2D(6, (5, 5), strides=1, padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (5, 5), strides=1, padding='same'))  
model.add(Activation('elu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (3, 3), strides=1, padding='same'))  
model.add(Activation('elu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (3, 3), padding='same'))  
model.add(Activation('elu'))
model.add(MaxPooling2D())
# Add flatten layer
model.add(Flatten())
# Add dense layers
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.20))

model.add(Dense(50, activation='elu'))
model.add(Dropout(0.20))

model.add(Dense(10, activation='elu'))
model.add(Dropout(0.5))

model.add(Dense(1))
# Print model summary
model.summary()
# Compile model
model.compile(loss='mse', optimizer='adam')
# Run model
model.fit(X_train, y_train, batch_size=1024, validation_split=0.2, shuffle=True, epochs=10)
# Save model
model.save('model.h5')
