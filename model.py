import csv
import cv2
import numpy as np
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dropout, Activation, Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split

samples = []
# Open and read csv file
with open('./data/set3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def resize_function(image):
    """Crop and Resize image.
    Args: image: image data
    Return: cropped and resized image 
    """
    image = image[60:140]
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    return  image

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    file_name = source_path.split('/')[-1]
                    current_path = './data/set3/IMG/' + file_name
                    image = cv2.imread(current_path)
                    if image is not None:
                        image = resize_function(image)
                        images.append(image)
                    else:
                        print('current path', current_path)

                center_angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.23 # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                angles.append(center_angle)
                angles.append(right_angle)
                angles.append(steering_right)
                augmented_images = []
                augmented_angles = []
                for image, angle in zip(images, angles):
                    augmented_images.append(image)
                    augmented_angles.append(angle)
                    # Flip images
                    augmented_images.append(cv2.flip(image, 1))
                    # Flip steering measurement
                    augmented_angles.append(angle*-1.0)


            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=1024)
validation_generator = generator(validation_samples, batch_size=1024)

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
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10)
# Save model
model.save('model.h5')
