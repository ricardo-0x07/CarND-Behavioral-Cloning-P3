import itertools
import csv
import cv2
import numpy as np
import os
from keras import backend as K
#if K.backend() != "tensorflow":
#    os.environ['KERAS_BACKEND'] = "tensorflow"
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
    image = image[60:140]
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    #image = image.reshape(45,160,1)
    return  image

with open('./data/set3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    low_steer = 0
    steerings = []
    #for line in itertools.islice(reader, 0, 15000):
    for line in reader:
        if float(line[3]) <= 0.000185:
            low_steer += 1
            if (low_steer % 3) == 0:
                steerings.append(line)
        else:
            steerings.append(line)
    for line in steerings:
    #for line in itertools.islice(reader, 0, 33000):  
      #print('line: ', line)
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.155 # this is a parameter to tune
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
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(64,64,3)))
#model.add(Lambda(lambda x: resize_function(x)))

model.add(Conv2D(24, (5, 5), strides=1, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('elu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))

model.add(Conv2D(36, (5, 5), strides=1, padding='same'))  
model.add(BatchNormalization(axis=1))
model.add(Activation('elu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))

model.add(Conv2D(48, (5, 5), strides=1, padding='same'))  
model.add(BatchNormalization(axis=1))
model.add(Activation('elu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))  
model.add(BatchNormalization(axis=1))
model.add(Activation('elu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), padding='same'))  
#model.add(BatchNormalization(axis=1))
#model.add(Activation('elu'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))

#model.summary()

model.add(Flatten())

#model.add(Dense(1164, activation='elu'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.25))

#model.add(Dense(100, activation='elu'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Dense(50, activation='elu'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='elu'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
#model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=512, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')
