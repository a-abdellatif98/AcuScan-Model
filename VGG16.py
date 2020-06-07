#importing the libraries 

import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
%matplotlib inline


#then i will red the data and do some processing on it 

# Get the VGG without the fulley conected layers 
vgg16 = VGG16(weights=None, include_top=True)

model = keras.models.Sequential()

for layer in vgg16.layers[0:19]:
    model.add(layer)


#here we had the problem the transition layer with some searching and asking people they have two opinions:
            #1- just ignore it and continue
            #2- it just a normal convolutional layer 
#so i choose the second one 

#the transition layer
model.add(keras.layers.Conv2D(16,(1,1),input_shape=(512,512,7)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(4, (1,1), strides=(1, 1), padding='valid',activation="relu"))

model.add(keras.layers.AvgPool2D(pool_size=2, strides=2))

#then adding globalMaxPoling
model.add(keras.layers.GlobalMaxPooling2D())

#then the sigmoid layer
model.add(Dense(13, activation='sigmoid'))

#and finally the loss function 
model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])