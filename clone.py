import itertools
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dropout, Activation, Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
lines = []
images = []
measurements = []
with open('./data/set3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in itertools.islice(reader, 0, 14338):
        #print('line: ', line)
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.5 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        # read in images from center, left and right cameras
        for i in range(3):
            source_path = line[i]
            file_name = source_path.split('/')[-1]
            current_path = './data/set3/IMG/' + file_name
            image = cv2.imread(current_path)
            images.append(image)
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
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5))

model.add(Conv2D(24, (4, 4)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(36, (4, 4)))  
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(48, (4, 4)))  
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2)))  
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2)))  
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.summary()

model.add(Flatten())

model.add(Dense(1164))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(100))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(50))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(10))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')
