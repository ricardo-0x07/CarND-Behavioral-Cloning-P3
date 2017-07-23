import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense


lines = []

with open('./data/set2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    file_name = source_path.split('/')[-1]
    current_path = './data/set2/IMG/' + file_name
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
input_shape = X_train.shape[1:]
#inp = Input(shape=input_shape)
#model.add(Flatten(input_shape=input_shape))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
