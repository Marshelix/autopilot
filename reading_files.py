import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os 
import sys
import pandas as pd

# have 3 formulations of dataset; [img,speed], [img,angle] and [img, angle, speed]

# decoding function and input to  networks 

def parse_1_label_shards(serialized): # shards are bunches of records similar to a batch
                                      # 
    ''' function calls on serialized data to revert back to images and labels 
       function will work for both  [img,speed] & [img,angle] since  key:label
       was provided in each case.'''
    
    features = \
        {'image': tf.io.FixedLenFeature([], tf.string),
         'label': tf.io.FixedLenFeature([], tf.int64)}
        
                # parsing serialized data to get a dict of data
    parsed_example = tf.io.parse_single_example(serialized = serialized,
                                                features = features)
    # parse_single_example treats bellow methods like they were acting on ONE individual 
    # example, but each will be iterator across and methods applied to
    image_raw = parsed_example['image']# image as raw bytes
    # decode raw bytes s.t becomes tensor with type
    image_decoded = tf.io.decode_raw(image_raw, tf.uint8) # "8" in "RGB/8" refers
                                                          # to 8 bits per colour channel; i.e. 8 for red,
                                                          # 8 for green and 8 for blue. So an "RGB/8" image 
                                                          # is 24 bits per pixel.
     # need float rather than int for graph computations to work 
    
    image = tf.cast(image_decoded, tf.float32)

    label = parsed_example['label']

    return image,label 

def parse_2_label_shards(serialized): # 
    ''' function calls on serialized data to revert back to images and labels 
        This function will do the parsing for 2 labels in one d; [img, angle, speed]'''
    features = \
        { 'image': tf.io.FixedLenFeature([], tf.string),
         'angle': tf.io.FixedLenFeature([], tf.int64),
         'speed': tf.io.FixedLenFeature([], tf.int64) }
    
    parsed_example = tf.io.parse_single_example(serialized = serialized,
                                                features = features)

    image_raw = parsed_example['image']
    image_decoded = tf.io.decode_raw(image_raw, tf.uint8) 
    image = tf.cast(image_decoded, tf.float32)
    
    label_angle = parsed_example['angle']
    label_speed = parsed_example['speed']

    
    return image, label_angle, label_speed 
#%%
BATCH_SIZE = 32
BUFFER = 1024 
file = 'trainingAngle.tfrecords'
number_of = 1

x, y = input_function(file,1, 1, 1)
#%%
def input_function(filenames = file,
                    train=False,
                    batch_size=BATCH_SIZE,
                    buffer_size=BUFFER,
                    num_of_labels = number_of):
    """ training and validation split: [70% 30%], 1660 total examples, """
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.
    
    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
  
    ds = tf.data.TFRecordDataset(filenames=filenames)
    if train == True:
        dataset_t = ds.take(1162) # 70% of 1660
        dataset_s = ds.skip(1162)
        
        dataset = dataset_t
   
    else: 
        dataset_t = ds.take(1162) # 70% of 1660
        dataset_s = ds.skip(1162)        
        dataset = dataset_s
    
     # Parse the serialized data in the TFRecords files.
    if num_of_labels == 1:
        dataset = dataset.map(parse_1_label_shards)
        print(dataset)
    if num_of_labels ==2:
        dataset = dataset.map(parse_2_label_shards)
        print(dataset) # recalls correct shape
        
        # need to split dataset into test train
        
        
    if train == True:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)
        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.
        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)
    
    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)
    # Create an iterator for the dataset and the above modifications.

    # Get the next batch of images and labels.
    if num_of_labels == 1:
        images_batch, labels_batch= next(iter(dataset))
        x = images_batch                     #=============================
        y = labels_batch
        return x, y
    
    if num_of_labels == 2:
        images_batch, labels_a_batch, labels_s_batch = next(iter(dataset))
        x = {'image': images_batch}
        a = labels_a_batch
        s = labels_s_batch
        return x, a, s
    
#%%

# these will return tensor objects to be passed onto y

def training_input_function():
    return input_function(filenames = file,
             train=True,
             batch_size=BATCH_SIZE,
             buffer_size=BUFFER,
             num_of_labels = number_of)
  
    
def testing_input_function():
    return input_function(filenames = file,
             train=False,
             batch_size=BATCH_SIZE,
             buffer_size=BUFFER,
             num_of_labels = number_of)
# =============================================================================
#         dataset = dataset.map( lambda image, label: (tf.image.convert_image_dtype(image, tf.float64),
#                                             label)).cache().take(5)

# train = train.map( lambda image, label: (tfa.image.translate(image, tf.random.uniform(shape=[2],minval=-5,
#                                                                                       maxval=5)), label))
# train = train_ds.map(lambda image, label: (tf.image.convert_image_dtype(image, tf.float64), label)).cache()
# train = train.map(lambda image, label: (tf.image.resize(image, [224, 224]), label))
# train = train.map(lambda image, label: (tf.image.random_flip_left_right(image), label))
# train = train.map(lambda image, label: (tfa.image.translate(image, tf.random.uniform(shape=[2], minval=-5, maxval=5)), label)).shuffle(100).batch(24).take(1)
# # note that translate is in tensorflow addons
# valid = validation_ds.map(lambda image, label: (tf.image.convert_image_dtype(image, tf.float64),label))
# valid = valid.map(lambda image, label: (tf.image.resize(image, [224, 224]), label)).shuffle(100).batch(24).take(1)
# test = test_ds.map(lambda image, label: (tf.image.convert_image_dtype(image, tf.float64), label)) 
# test = test.map(lambda image, label: (tf.image.resize( image, [224,224]), label)).batch(100).take(1)
# =============================================================================


#%%
# 

model = tf.keras.Sequential([tf.keras.layers.Conv2D( 24, (5,5), strides=(2,2), input_shape=(320, 240, 3), activation='relu'),
                             tf.keras.layers.Conv2D( 36, (5,5), strides=(2,2), activation = 'relu'),
                             tf.keras.layers.Conv2D( 48, (5,5), strides=(2,2), activation='relu'),
                             tf.keras.layers.Conv2D( 64, (3,3), activation='relu'), 
                             tf.keras.layers.Dropout( 0.25),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dropout(0.25),
                             tf.keras.layers.Dense(100, activation='relu'),
                             tf.keras.layers.Dense(50, activation='relu'),
                             tf.keras.layers.Dense(10, activation='relu'),
                             tf.keras.layers.Dense(1, activation='elu'), # predicting turning angle
                             #tf.keras.losses.mean_squared_error(label_val[:,0][:,np.newaxis])
                             ])
learningRate = 0.001                            
opt = tf.optimizers.Adam(lr=learningRate) # lr is learning rate
model.compile(loss='mse', optimizer=opt)
model.build()
model.summary()
#%%

model.fit(training_input_function(), epochs=5)

#%%
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
    use_multiprocessing=False, **kwargs
)











### dont split the datasets!!!!
#%%
ds_path = (r'/Users/Akin/Documents/PostGraduate Machine Learning'
                 ' in Science/Machine Learning in Science Part 2 Module/'
                 'autonomousCarProject/tensor_datasets/')

 # need subdirectory insides this one for flow_from..
# buiilding pip line for training, validating and testing 

target_names =['feature', 'label']
ds_speed = tf.keras.utils.get_file(fname=ds_path+'ds_speed', origin=ds_path+'ds_speed')


#%%

dslv_iter = label_flow.flow(x=ds_v  ,y = ds_label_train,
                            batch_size=BATCH_SIZE)

dslv = dslv_iter.data.Dataset.from_generator(lambda : dslv_iter,
                                             output_types = (tf.float32, tf.float32),
                                             output_shape= ([BATCH_SIZE,1,1]))

#%%
data_gen = ImageDataGenerator(rescale=1./255)
trainIter = data_gen.flow_from_directory(train_path,
                                       target_size=(240, 320),
                                       color_mode="rgb",
                                       classes=None,
                                       batch_size=BATCH_SIZE)

ds_t = tf.data.Dataset.from_generator(lambda: trainIter,
                                      output_types= ( tf.float64, tf.float64),
                                      output_shapes = ([BATCH_SIZE, 240, 320,3], [BATCH_SIZE,1,1]))
#%%

data_gen = ImageDataGenerator(rescale=1./255)
validIter =data_gen.flow_from_directory(valid_path, # notice from directory
                                       target_size=(240,320),
                                       color_mode="rgb",
                                       class_mode=None
                                       batch_size=BATCH_SIZE)

ds_v = tf.data.Dataset.from_generator( lambda: validIter, # note that lambda acts on the iterator defined above
                                    output_types=(tf.float64, tf.float64),
                                    output_shapes = ([BATCH_SIZE, 240,320,3],[BATCH_SIZE,1,1]))

#%%
ds_train = tf.data.Dataset.zip((ds_t, ds_label_train))
ds_valid = tf.data.Dataset.zip((ds_v, ds_label_val))

#%%


#%%





















#%%