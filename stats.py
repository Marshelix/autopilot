# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:08:08 2020

@author: marti
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
import logging
import sys

logging.basicConfig(level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("car_log.txt"),
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger("CarAi")
logger.debug("File created. Starting Log.")
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
    logger.debug(str(len(fnames_full)) + " names found.")
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
    logger.info("Loading data from "+dirname)
    fnames_full = glob.glob(dirname,recursive = True)
    logger.info(str(len(fnames_full)) + " names found")
    for fname in fnames_full:
        fname_split = fname.split("\\")
        logger.debug("Loading {}".format(fname))
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

def any_word_in_string(string,words):
    ret = False
    if words == None:
        return True
    for w in words:
        ret = ret or (w in string)
    return ret

def load_dataset(resize_percent =  60,restrict_folders = None,restrict_angles = None, restrict_speeds = None):
    images = []
    targets = []
    dirname = "./data/**/*.png"
    logger.info("Loading data from "+dirname)
    fnames_full = glob.glob(dirname,recursive = True)
    logger.info(str(len(fnames_full)) + " names found")
    for fname in fnames_full:
        fname_split = fname.split("\\")
        
        name = fname_split[2]
        speed = (float(name.split(".")[0].split("_")[2]))/35.0
        angle = (float(name.split(".")[0].split("_")[1])-50.0)/80.0
        img = pltimg.imread(fname)
        width = int(img.shape[1] * resize_percent / 100)
        height = int(img.shape[0] * resize_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        folder_fit = restrict_folders == None or (any_word_in_string(fname, restrict_folders))
        angle_fit = restrict_angles == None or (float(angle) in restrict_angles)
        speed_fit = restrict_speeds == None or (float(speed) in restrict_speeds)
        if folder_fit and angle_fit and speed_fit:
            logger.debug("Loading {}".format(fname))
            images.append(resized)
            targets.append([float(angle),float(speed)])
    return images, targets

own_set_foldertags = ["clear-ring","clear-t","noise","objects-ring","t-turn"]
luke_tags = ["dataset"]
josh_tags = ["capture"]
imgs, targets = load_dataset(resize_percent = 60, restrict_folders = own_set_foldertags)

own_angle_hist = np.asarray(targets)[:,0]
own_speed_hist = np.asarray(targets)[:,1]


plt.figure()
plt.hist(own_angle_hist)
plt.title("Histogram of angles in Group Max, Akin, Martin taken data")
plt.savefig("MAMAngleHistogram.png")

plt.figure()
plt.hist(own_speed_hist)
plt.title("Histogram of speeds in Group Max, Akin, Martin taken data")
plt.savefig("MAMspeedHistogram.png")