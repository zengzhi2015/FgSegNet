#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:22:22 2017

@author: longang
@Comments by Zhi Zeng
"""

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

"""
Note:
	autoreload ensures that we can run a imported function, then, change the function in an editor and the changed function can be reloaded
"""

"""
Note:
	get_ipython().magic('zzz') equals to %zzz
"""

import numpy as np
import tensorflow as tf
import random as rn
import os

# =============================================================================
#  For reprodocable results
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
# Environment variables are accessed through os.environ
# If PYTHONHASHSEED is set to an integer value, it is used as a fixed seed for generating the hash() of the types covered by the hash randomization.
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# ConfigProto is used to configurate sesson()
# For both configuration options, if they are unset or set to 0, will default to the number of logical CPU cores. Testing has shown that the default is effective for systems ranging from one CPU with 4 cores to multiple CPUs with 70+ combined logical cores. A common alternative optimization is to set the number of threads in both pools equal to the number of physical cores rather than logical cores.
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras, glob
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from sklearn.utils import compute_class_weight
from FgSegNetModule import FgSegNetModule
from keras.utils.data_utils import get_file

# =============================================================================
# Few frames, load into memory directly
# =============================================================================
def generateData(train_dir, dataset_dir, scene):
    
    # It is weired that the parameter 'scene'is not used in the function
    
    void_label = -1. # The masked regions will be labeled as -1
    X_list = []
    Y_list = []
    
    # Given ground-truths, load training frames
    # ground-truths end with '*.png'
    # training frames end with '*.jpg'
    
    # scan over FgSegNet_dataset for groundtruths
    for root, _, _ in os.walk(train_dir):
        # Walk though the tree (root, dirs, files)
        gtlist = glob.glob(os.path.join(root,'*.png'))
        # list all *.png files
        if gtlist:
            Y_list = gtlist
        # I think the following two lines should be add
        else:
            Y_list += gtlist
    
    # scan over CDnet2014_dataset for .jpg files
    for root, _, _ in os.walk(dataset_dir):
        inlist = glob.glob(os.path.join(root,'*.jpg'))
        if inlist:
            X_list = inlist
        # I think the following two lines should be add
        else:
            X_list += inlist
    
    # filter matched files        
    # brutal force match
    X_list_temp = []
    for i in range(len(Y_list)):
        Y_name = os.path.basename(Y_list[i])
        Y_name = Y_name.split('.')[0]
        Y_name = Y_name.split('gt')[1]
        for j in range(len(X_list)):
            X_name = os.path.basename(X_list[j])
            X_name = X_name.split('.')[0]
            X_name = X_name.split('in')[1]
            if (Y_name == X_name):
                X_list_temp.append(X_list[j])
                break
    X_list = X_list_temp
    # The following line is negligable
    # del X_list_temp, inlist, gtlist
    
    # process training images
    X = []
    Y = []
    # Load all images and modify the lables
    for i in range(0, len(X_list)):
        x = kImage.load_img(X_list[i])
        x = kImage.img_to_array(x)
        X.append(x)
        # what is the type of X ?
        # Is the range of an element in X [0, 255] or [0,1]?
        
        x = kImage.load_img(Y_list[i], grayscale = True)
        x = kImage.img_to_array(x)
        shape = x.shape
        x /= 255.0
        x = x.reshape(-1)
        idx = np.where(np.logical_and(x>0.25, x<0.8))[0] # find non-ROI
        if (len(idx)>0):
            x[idx] = void_label # void_label = -1
            # After this operation, elements in Y can be only one of [0.0, 1.0, -1.0]
        x = x.reshape(shape)
        x = np.floor(x)
        Y.append(x)
    # The following line is negligable
    # del Y_list, X_list, x, idx
    X = np.asarray(X) # convert X to np.array
    Y = np.asarray(Y)
    
    # Shuffle the training data
    idx = list(range(X.shape[0])) # np.arange(X.shape[0])
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    # The following line is negligable
    # del idx
    
    # Image Pyramid
    scale1 = X
    # del X
    scale2 = []
    scale3 = []
    for i in range(0, scale1.shape[0]):
       pyramid = tuple(pyramid_gaussian(scale1[i]/255., max_layer=2, downscale=2))
       scale2.append(pyramid[1]) # 2nd scale
       scale3.append(pyramid[2]) # 3rd scale
       del pyramid
    scale2 = np.asarray(scale2)
    scale3 = np.asarray(scale3)
    # I think the following line should added:
    scale1 = X/255.
    print (scale1.shape, scale2.shape, scale3.shape)

    # compute class weights
    cls_weight_list = []
    for i in range(Y.shape[0]):
        # For each groundtruth image, the auther compute the class weights. These class weights are stored in a list
        # I do not know why the author need this.
        y = Y[i].reshape(-1) # flattern Y
        idx = np.where(y!=void_label)[0] # for unmasked pixels
        if(len(idx)>0):
            y = y[idx] # exclude masked pixels
        lb = np.unique(y) #  0., 1.0
        cls_weight = compute_class_weight('balanced', lb , y)
        # cls_weight = n_samples/(n_classes*np.bincount(y))
        # np.bincount count the num of occurence of each non-neg ints
        class_0 = cls_weight[0]
        class_1 = cls_weight[1] if len(lb)>1 else 1.0
        
        cls_weight_dict = {0:class_0, 1: class_1}
        cls_weight_list.append(cls_weight_dict)
    # del y, idx
    cls_weight_list = np.asarray(cls_weight_list)
    # Three kinds of inputs, one outputs, and one weight list
    return [scale1, scale2, scale3, Y, cls_weight_list]
    
def train(results, scene, mdl_path, log_dir, vgg_weights_path):
    
    # results: [scale1, scale2, scale3, Y, cls_weight_list]
    # scene: string for the name of the scene (link 'baseline')
    # mdl_path: store checkpoints of tensorflow
    # log_dir: used for tensorboard
    # vgg_weights_path: path for the vgg16 model
    
    img_shape = results[0][0].shape
    # results[0] = scale1 (numbers,width,height,channels)
    model = FgSegNetModule(lr, reg, img_shape, scene, vgg_weights_path)
    # it seems that lr (learning rate) and reg (regulizer weight) are defined outside
    model = model.initModel()
    
    tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    chk = keras.callbacks.ModelCheckpoint(mdl_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=num_patience, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    model.fit([results[0], results[1], results[2]], results[3], validation_split=0.2, epochs=epoch, batch_size=batch_size, 
                       callbacks=[redu, chk, tb], verbose=1, class_weight=results[4], shuffle = True)

    del model, results, tb, chk, redu

dataset = {
            'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
            'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
            'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
            'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
            'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
            'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
            'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
            'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
}

# =============================================================================
num_frames = 50 # either 50 or 200 frames
reduce_factor = 0.1
num_patience = 6
lr = 1e-4
reg=5e-4
epoch = 60 if num_frames==50 else 50 # 50f->60epochs, 200f->50epochs
batch_size = 1
# =============================================================================

# Example: (free to modify)

# FgSegNet/FgSegNet/FgSegNet.py
# FgSegNet/FgSegNet/FgSegNetModule.py
# FgSegNet/FgSegNet_dataset2014/...
# FgSegNet/CDnet2014_dataset/...
# FgSegNet/models/...
# FgSegNet/logs/...


main_dir = 'FgSegNet'

main_mdl_dir = os.path.join(main_dir,'models', 'f' + str(num_frames))
main_log_dir = os.path.join(main_dir,'logs', 'f' + str(num_frames))
vgg_weights_path = os.path.join(main_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
if not os.path.exists(vgg_weights_path):
    # keras func
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                file_hash='6d6bbae143d832006294945121d1f1fc')

for category, scene_list in dataset.items():
    
    mdl_dir = os.path.join(main_mdl_dir, category)
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)
        
    log_dir = os.path.join(main_log_dir, category)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    for scene in scene_list:
        print ('Training ->>> ' + category + ' / ' + scene)
        
        # training frame path and dataset2014 path
        # '..' up dir 1 level 
        train_dir = os.path.join('..', 'FgSegNet_dataset2014', category, scene + str(num_frames))
        dataset_dir = os.path.join('..', 'CDnet2014_dataset', category, scene)
            
        mdl_path = os.path.join(mdl_dir, 'mdl_' + scene + '.h5')

        results = generateData(train_dir, dataset_dir, scene)
        train(results, scene, mdl_path, log_dir, vgg_weights_path)
        del results
