from random import shuffle
import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as data_file:
	reader = csv.reader(data_file)
	next(reader, None)
	for line in reader:
		lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
				left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
				right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
				center_image = cv2.imread(center_name)
				left_image = cv2.imread(left_name)
				right_image = cv2.imread(right_name)
				center_angle = float(batch_sample[3])
				left_angle = center_angle + 0.2
				right_angle = center_angle - 0.2
				images.append(center_image)
				angles.append(center_angle)
				images.append(left_image)
				angles.append(left_angle)
				images.append(right_image)
				angles.append(right_angle)
				if abs(center_angle) > 0.1:
					center_image_mirrored = cv2.flip(center_image, 1)
					center_angle_mirrored = -center_angle
					images.append(center_image_mirrored)
					angles.append(center_angle_mirrored)
				if abs(left_angle) > 0.1:
					left_image_mirrored = cv2.flip(left_image, 1)
					left_angle_mirrored = -left_angle
					images.append(left_image_mirrored)
					angles.append(left_angle_mirrored)
				if abs(right_angle) > 0.1:
					right_image_mirrored = cv2.flip(right_image, 1)
					right_angle_mirrored = - right_angle
					images.append(right_image_mirrored)
					angles.append(right_angle_mirrored)
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')
