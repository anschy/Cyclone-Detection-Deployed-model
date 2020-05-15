from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn import preprocessing
from keras.preprocessing.image import load_img, img_to_array
import pickle


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join

img_width = 150
img_height = 150

train_data_dir = 'C://Users//KIIT//Desktop//project cyclone detection//train'
validation_data_dir = 'C://Users//KIIT//Desktop//project cyclone detection//validation'
train_samples = 120
validation_samples = 30
epochs = 5
batch_size = 20

# Check for TensorFlow or Thieno
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = Sequential()


model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


import keras
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=.0001),metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

imgs, labels = next(train_generator)


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples)

from keras.preprocessing.image import load_img, img_to_array



#predicting a cyclonic image.....if 1 , cyclone(true)
model.predict(np.array([image.img_to_array(load_img('C://Users//KIIT//Desktop//project cyclone detection//test//1001.jpg').resize((150,150)))/255
]))


model.predict(np.array([image.img_to_array(load_img('C://Users//KIIT//Desktop//project cyclone detection//train//NON-CYCLONE//images (13).jfif').resize((150,150)))/255
]))


pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))




    



