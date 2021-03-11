import numpy as np 
import pandas as pd

#We are using keras as our base package to build our binary classifier. 
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.utils.vis_utils import plot_model
from keras.callbacks import  EarlyStopping

import os
import gc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

def img_gen(train_path_variable, val_path_variable, test_path_variable):

    train_preprocess = keras.preprocessing.image.ImageDataGenerator(
                                            rescale = 1./255,
                                            zoom_range = 0.2,
                                            shear_range = 0.2,
                                            horizontal_flip = True
                                        )

    train = train_preprocess.flow_from_directory(
                                            train_path_variable,
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'binary'
                                        )
    validation_preprocess = keras.preprocessing.image.ImageDataGenerator(
                                            rescale=1./255
                                        )

    val = validation_preprocess.flow_from_directory(
                                            val_path_variable,
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'binary'
    )

    test_preprocess = keras.preprocessing.image.ImageDataGenerator(
                                            rescale=1./255
                                        )

    test = test_preprocess.flow_from_directory(
                                            test_path_variable,
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'binary'
                                            )
    return train, val, test


