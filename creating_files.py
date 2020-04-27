import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os 
import sys
import pandas as pd
#%%
# load the x_train_and_x_label.spydata into variable exploreR
#
# =============================================================================
# gpu = tf.config.experimental.list_physical_devices('GPU')
# if len(gpu)>1:
#     tf.congfig.experimental.set_virtual_device_configuration(gpu[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
# for gpu config not to worry about 
# =============================================================================

no_of_speed_classes = list(set(x_label[:,1].astype(int)))  #
no_of_angle_classes = list(set(x_label[:,0]))              # betting list of unique values; i.e. classes

negative_speed_instances = np.where(x_label==-1) 
# to be dropped or altered to positive? -> changed to 0
x_label[negative_speed_instances[0][1],:] = 0
x_label[negative_speed_instances[0][0],:] = 0
# one hot encoding for speed and angle, number classes with integers
speed_indices = [0, 1] # class number/label
angle_indices = np.arange(0,18).tolist() # class number/label
speed_depth = 2   # for one hot vector encoding 
angle_depth = 17  #   
cls_labels_speed = tfds.features.ClassLabel( names=['stop','drive'])         #
cls_labels_angle = tfds.features.ClassLabel( names=map(str, np.arange(0,18)))# not necessary
# integer class numbers -> lables, class -> one_hot_encoding
#%%
cls_speed = tf.one_hot(speed_indices, speed_depth, dtype=tf.float64, name='Speed')
cls_angle = tf.one_hot(angle_indices,angle_depth, dtype=tf.float64, name='Angle')
#%%
training_data_path = (r'\Users\Akin\Documents\PostGraduate Machine Learning in Science\Machine Learning in Science Part 2 Module\autonomousCarProject\tensor_datasets\train\train')
image_paths = r'\Users\Akin\Documents\PostGraduate Machine Learning in Science\Machine Learning in Science Part 2 Module\autonomousCarProject\tensor_datasets\train\train'
# validation_data_path =(r'\Users\Akin\Documents\PostGraduate Machine Learning in Science\Machine Learning in Science Part 2 Module\autonomousCarProject\tensor_datasets\validation')
# getting paths of all training data and ordering it such that labels can be align, merged valid and training can split after pipline processing 
#image_paths_train = [f for f in os.listdir() if os.listdir(training_data_path+"\\train") if os.path.isfile(os.path.join(my_path, f))]
image_paths_train = [ f for f in os.listdir(training_data_path) if os.path.isfile( os.path.join(training_data_path, f) )]

image_paths_train = sorted(image_paths_train, key=lambda x: int(x.partition('n')[2].partition('.')[0]))
# image_paths_valid = [ f for f in os.listdir() if os.listdir(validation_data_path+"\valid") if os.path.isfile(os.path.join(my_path, f))]
#%%
path_to_angle_data = (r'\Users\Akin\Documents\PostGraduate Machine Learning in Science\Machine Learning in Science Part 2 Module\autonomousCarProject\tensor_datasets\train\train_angle_classes.csv')

path_to_speed_data = (r'\Users\Akin\Documents\PostGraduate Machine Learning in Science\Machine Learning in Science Part 2 Module\autonomousCarProject\tensor_datasets\train\train_speed_classes.csv')
labels_names_angle = np.loadtxt(path_to_angle_data, delimiter=',') * 1000 # need to normalise the angles after, required to be int 


labels_names_speed = np.loadtxt(path_to_speed_data, delimiter=',',  skiprows=1)
# needed long for encoding, has to be int for one hot encoding
# had to drop the first element of angle labels since it was picking up ï»¿
labels_names_speed = np.insert(labels_names_speed,0,0 )
#%% 
labels_names_angle = list(map(int, labels_names_angle))
labels_names_speed = list(map(int, labels_names_speed))
#

#%%
print("Size of :")
print("- Training set: \t \t{}".format(len(image_paths_train)))

#%%#
# file path for tfrecord
tfrecords_path_saving = (r'\Users\Akin\Documents\PostGraduate Machine Learning in Science\Machine Learning in Science Part 2 Module\autonomousCarProject\tensor_datasets\train\trainingAngle1.tfrecords')

# =============================================================================
#                                                          file name of tf record file
path_tfrecords_train = os.path.join(tfrecords_path_saving, "trainingAngle.tfrecords")
# 
# =============================================================================

# reporting function for when converting; convert()
def print_progress(count, total):
    percent_complete = float(count)/total
    msg = "\r -Progress: {0:.1%}".format(percent_complete) 
    sys.stdout.write(msg)
    sys.stdout.flush()


# function to help store integer and bytes to be saved in tf records, need
# to define additonal wrap for whatever labels  and content(images) in dataset
#  do not have type already specified

# make sure whatever the _wrap used below matches the dtype; above is int  for labels 
# check warp and will be train.Feature(=>int64<= = tf.train.=>Int64List<=(....))
def _wrap_int64(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
def _wrap_bytes(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


# function reads ims of disk, writes with class labels to TF_records file. 
# Loads and decodes ims to np.array and stores raw bites in TF_records
# terminal such as records, shards streams used throughout industry; look at AWS 

def convert_1_label( image_paths, labels,  out_path):
    
    
    ''''' need directory to be folder with training examples in. Function 
        binarizes data and save to format easy to process with net, 
        use tf.data.TFRecordDataset, where all the data prep, augmenting and so
        on can be done in the following functions
        
        to change the file name  of tensorflow record file, look above ============
        '''
    print("converting"+ out_path)
    num_images = len(image_paths)
    with tf.io.TFRecordWriter(out_path) as writer:
        for i ,(path, label) in enumerate(zip(image_paths, labels)):
            
            # only needing i for print report, thats it!
            #=============================
            print( (path, label) )
            #=============================
            print_progress(count=i, total = num_images-1)
            
            img = plt.imread(path)
            img_bytes = img.tostring_rgb()
            
            # dict of tfrecords
            data = { 'image': _wrap_bytes(img_bytes),
                     'label': _wrap_int64(label) } # using 'label' key for both ds 
                                                   # where only 1 class associated to
                     
            
            feature = tf.train.Features(feature=data) # tensor flow features
            example = tf.train.Example(features=feature) # tensorflow example
            serialized = example.SerializeToString() # serialise 
            
            writer.write(serialized)

def convert_2_label(image_paths, labels_A, labels_S, out_path):
    
    
    ''''' need directory to be folder with training examples in. Function 
        binarizes data and save to format easy to process with net, 
        use tf.data.TFRecordDataset, where all the data prep, augmenting and so
        on can be done in the following functions
        
        to change the file name  of tensorflow record file, look above ============
        '''
    print("converting"+ out_path)
    num_images = len(image_paths)
    with tf.io.TFRecordWriter(out_path) as writer:
        for i ,(path, label1, label2) in enumerate(zip(image_paths, labels_A, labels_S)):
            
            # only needing i for print report, thats it!
            #=============================
            print( (path, label1, label2))
            #=============================
            print_progress(count=i, total = num_images-1)
            
            img = plt.imread(path)
            img_bytes = img.tostring()
            
            # dict of tfrecords
            data ={'image': _wrap_bytes(img_bytes),
                    'angle': _wrap_int64(label1), # can add addition labels
                    'speed': _wrap_int64(label2)} # notice, it is NOT label[i]
                     
            
            feature = tf.train.Features(feature=data) # tensor flow features
            example = tf.train.Example(features=feature) # tensorflow example
            serialized = example.SerializeToString() # serialise 
            
            writer.write(serialized)
            
#%%
# note the use of inter class numbers instead of one hot vector
# not the image_paths_train is a sorted version, needed for aligning labels and examples

# function call will  produce serialised ds for two classes
convert_1_label(image_paths=image_paths_train,
                labels = labels_names_angle,
                out_path =tfrecords_path_saving)
#%%
convert_2_label(image_paths = image_paths_train,
                labels_A = labels_names_angle,
                labels_S = labels_names_speed,
                out_path = path_tfrecords_train)
#%%
#