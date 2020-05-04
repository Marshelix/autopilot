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
import tensorflow.keras.preprocessing.image as imgprep

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

#Adam stuff
import pandas as pd

'''
    File to train a simple cnn on image data gathered with pi-car
    Model built and trained using keras, transferrable to tensorflow lite
'''

model_output_dir = "./models/"

mode = "**"

results = {}

def load_test_dataset(resize_percent = 60):
    dirname = "./test_data/test_data/**.png"
    fnames_full = glob.glob(dirname, recursive = True)
    print(str(len(fnames_full)) + " names found.")
    images = [[] for i in range(len(fnames_full))]  #override image according to name
    for fname in fnames_full:
        img = pltimg.imread(fname)
        width = int(img.shape[1] * resize_percent / 100)
        height = int(img.shape[0] * resize_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
        name_idx = int(fname.split("\\")[1].split(".")[0])-1
        images[name_idx] = resized
    return np.asarray([images])

def load_adam_dataset(resize_percent = 60):
    train_df = pd.read_csv("training_norm.csv")
    train_df.index = train_df["image_id"]
    angles = train_df.angle
    speed = train_df.speed
    targets = []
    images = []
    dirname = "./training_data/**/*.png"
    print("Loading data from "+dirname)
    fnames_full = glob.glob(dirname,recursive = True)
    print(str(len(fnames_full)) + " names found")
    for fname in fnames_full:
        fname_split = fname.split("\\")
        #print("Loading {}".format(fname))
        name = fname_split[2]
        img = pltimg.imread(fname)
        width = int(img.shape[1] * resize_percent / 100)
        height = int(img.shape[0] * resize_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        images.append(resized)
        targets.append([float(train_df.angle[int(name.split(".")[0])]),float(train_df.speed[int(name.split(".")[0])])])
    return images,targets

adam_images,adam_targets = load_adam_dataset()

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
        #print("Loading {}".format(fname))
        speed = (float(name.split(".")[0].split("_")[2]))/35.0
        angle = (float(name.split(".")[0].split("_")[1])-50.0)/80.0
        img = pltimg.imread(fname)
        width = int(img.shape[1] * resize_percent / 100)
        height = int(img.shape[0] * resize_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        
        images.append(resized)
        targets.append([float(angle),float(speed)])
        
    return images, targets


imgs, targets = load_dataset()
imgs = imgs + adam_images
targets = targets + adam_targets
in_shape = imgs[0].shape
X_train, X_valid, y_train, y_valid = train_test_split( imgs, targets, test_size=0.2)
print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

#Define Model:
# MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=in_shape, 
    include_top=False, weights='imagenet')
base_model.trainable = False
model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(2)
])
print(model.summary())


# InceptionV3
base_model = tf.keras.applications.InceptionV3(input_shape=in_shape, 
    include_top=False, weights='imagenet')
base_model.trainable = False
model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(2)
])
print(model.summary())


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
    
    optimizer = Adam(lr = 1e-3)
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

aug = imgprep.ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# saves the model weights after each epoch if the validation loss decreased
class PredictionCallback(tf.keras.callbacks.Callback):
    """On an improved epoch, make predictions for the assessment data
    
    Arguments:
        test_images: the assessment data you want predictions for every
        time the model improves.
        global_best: the best MSE loss score across all your models.
    """

    def __init__(self, test_images, global_best=np.Inf):
        super(PredictionCallback, self).__init__()

        self.test_images = test_images
        self.global_best = global_best

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        if np.less(current, self.best):
            self.global_best = self.best
            self.make_prediction()

    def make_prediction(self):
        colnames = ["angle","speed"]
        prediction = model.predict(test_images[0])
        df = pd.DataFrame(prediction, columns = colnames)
        df["image_id"] = range(1, len(test_images[0])+1)
        df = df.reindex(columns = ["image_id","angle","speed"])
        df.index = df["image_id"]
        df = df[["angle","speed"]]
        #write
        df.to_csv("prediction.csv")

        
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_output_dir,'AI_driver_check_all.h5'), verbose=1, save_best_only=True)
prediction_callback = PredictionCallback(test_images=load_test_dataset())
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
history = pick_a_tl_model.fit_generator(aug.flow(np.array(X_train),y_train,batch_size = 100),
                              steps_per_epoch=300,
                              epochs=500,
                              validation_data = image_data_gen(X_valid,y_valid,100),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1,
                              callbacks=[checkpoint_callback, es_callback])
# always save model output as soon as model finishes training
name = os.path.join(model_output_dir,'AI_driver_all.h5')
model.save(name)
plt.figure()
plt.plot(history.history['loss'],color='blue')
plt.legend(["training loss"])
plt.figure()
plt.plot(history.history['val_loss'],color='red')
plt.legend([ "validation loss"])
results[mode] = [history.history['loss'][-1],history.history['val_loss'][-1]]


#create prediction
# colnames = ["angle","speed"]
# test_images = load_test_dataset()
# prediction = model.predict(test_images[0])
# df = pd.DataFrame(prediction, columns = colnames)
# df["image_id"] = range(1,len(test_images)+1)
# df = df.reindex(columns = ["image_id","angle","speed"])
# df.index = df["image_id"]
# df = df[["angle","speed"]]
# #write
# df.to_csv("prediction.csv")


#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#conv_name = os.path.join(model_output_dir,"converted_model.tflite")
#open(conv_name, "wb").write(tflite_model)

