# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:15:37 2020

@author: Martin Sanner
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# Data loading
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

import random
import os

# sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

'''
    File to train a simple cnn on image data gathered with pi-car
    Model built and trained using keras, transferrable to tensorflow lite
'''

model_output_dir = "./models/"

mode = "**"

results = {}

def load_dataset(resize_percent =  60):
    images = []
    targets = []
    dirname = "./data/**/*.png"
    print("Loading data from "+dirname)
    fnames_full = glob.glob(dirname,recursive = True)
    print(str(len(fnames_full)) + " names found")
    for fname in fnames_full:
        fname_split = fname.split("\\")
        
        name = fname_split[2]
        speed = name.split(".")[0].split("_")[2]
        angle = name.split(".")[0].split("_")[1]
        img = pltimg.imread(fname)
        width = int(img.shape[1] * resize_percent / 100)
        height = int(img.shape[0] * resize_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        
        images.append(resized)
        targets.append([float(angle),float(speed)])
        
    return images, targets




imgs, targets = load_dataset()
in_shape = imgs[0].shape
X_train, X_valid, y_train, y_valid = train_test_split( imgs, targets, test_size=0.2)
print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

#Define Model:
def Model():
    model = Sequential()
    model.add(BatchNormalization(input_shape = in_shape))
    model.add(Conv2D(12,(5,5),strides = (2,2),input_shape = in_shape,activation = "elu"))
    model.add(Conv2D(30,(3,3),strides = (2,2), activation = "elu"))
    model.add(Conv2D(40,(5,5),strides = (2,2), activation = "elu"))
    model.add(Conv2D(64,(5,5),strides = (2,2), activation = "elu"))
    
    #fully connected output
    model.add(Flatten())
    model.add(Dense(100, activation = "elu"))
    model.add(Dense(50, activation = "elu"))
    model.add(Dense(20, activation = "elu"))
    model.add(Dense(2,activation = "elu"))
    
    optimizer = Adam(lr = 1e-2)
    model.compile(loss = "mse", optimizer = optimizer)
    return model

model = Model()
print(model.summary())


def image_data_gen(imgs, targets,batch_size):
    while True:
        batch_images = []
        batch_targets = []
        for i in range(batch_size):
            random_index = random.randint(0,len(imgs)-1)
            img = imgs[random_index]
            target = targets[random_index]
            batch_images.append(img)
            batch_targets.append(target)
        yield(np.asarray(batch_images), np.asarray(batch_targets))

# saves the model weights after each epoch if the validation loss decreased

        
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_output_dir,'AI_driver_check_all.h5'), verbose=1, save_best_only=True)
    
history = model.fit_generator(image_data_gen( X_train, y_train, batch_size=100),
                              steps_per_epoch=300,
                              epochs=50,
                              validation_data = image_data_gen( X_valid, y_valid, batch_size=100),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1,
                              callbacks=[checkpoint_callback])
# always save model output as soon as model finishes training
name = os.path.join(model_output_dir,'AI_driver_all.h5')
model.save(name)
plt.figure()
plt.plot(history.history['loss'],color='blue')
plt.plot(history.history['val_loss'],color='red')
plt.legend(["training loss", "validation loss"])
results[mode] = [history.history['loss'][-1],history.history['val_loss'][-1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
conv_name = os.path.join(model_output_dir,"converted_model.tflite")
open(conv_name, "wb").write(tflite_model)

