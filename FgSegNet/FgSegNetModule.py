#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:57:12 2017

@author: longang
@comments by Zhi Zeng
"""

import keras

from keras.models import Model
from keras.layers import Activation, Input, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D, MyUpSampling2D
"""
Note: The module MyUpSampling2D is written by the author.
"""
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

class FgSegNetModule(object):
    
    def __init__(self, lr, reg, img_shape, scene, vgg_weights_path):
        self.lr = lr # Is this the learning rate ??
        self.reg = reg # The weight for the l2 regularizer for the Tconv
        self.img_shape = img_shape # the shape of the image ??
        self.scene = scene # ???
        self.vgg_weights_path = vgg_weights_path # the path for the vgg_weights

    def VGG16(self, x): 
    """
    Note: This defines the structure of the vgg16. Note that each layer has a unique name in the definition.
    """
        
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        """
        Note:
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the  inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
        """
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)
        
        return x
        
    def transposedConv(self, x):
        
        # block 5
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block5_tconv1', kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block5_tconv2')(x)
        x = Conv2DTranspose(512, (1, 1), activation='relu', padding='same', name='block5_tconv3')(x)
        """
        Note:
        Usage of regularizers

        Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.

        The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers Dense, Conv1D, Conv2D and Conv3D have a unified API.

        These layers expose 3 keyword arguments:

        kernel_regularizer: instance of keras.regularizers.Regularizer
        bias_regularizer: instance of keras.regularizers.Regularizer
        activity_regularizer: instance of keras.regularizers.Regularizer
        Example

        from keras import regularizers
        model.add(Dense(64, input_dim=64,
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))
        Available penalties

        keras.regularizers.l1(0.)
        keras.regularizers.l2(0.)
        keras.regularizers.l1_l2(0.)
        """
        
        # block 6
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block6_tconv1', kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block6_tconv2')(x)
        x = Conv2DTranspose(256, (1, 1), activation='relu', padding='same', name='block6_tconv3')(x)
        
        # block 7
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block7_tconv1', kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block7_tconv2')(x)
        x = Conv2DTranspose(128, (1, 1), activation='relu', padding='same', name='block7_tconv3')(x)
        
        # block 8
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block8_conv1', kernel_regularizer=regularizers.l2(self.reg))(x)
        """
        Note: 
        For each of the Tconv block, there is a regularizer. I am not sure why the regularizer should be used here?
        """
        
        # block 9
        x = Conv2DTranspose(1, (1, 1), padding='same', name='block9_conv1')(x)
        x = Activation('sigmoid')(x)
        
        return x

    def initModel (self):
        h, w, d = self.img_shape # height width dimension
        
        input_1 = Input(shape=(h, w, d), name='ip_scale1')
        vgg_layer_output = self.VGG16(input_1)
        shared_model = Model(inputs=input_1, outputs=vgg_layer_output, name='shared_model')
        shared_model.load_weights(self.vgg_weights_path, by_name=True)
        """
        Note:
        model.load_weights(filepath, by_name=False): loads the weights of the model from a HDF5 file (created by save_weights). By default, the architecture is expected to be unchanged. To load weights into a different architecture (with some layers in common), use by_name=True to load only those layers with the same name.
        """
        
        unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']
        for layer in shared_model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
        """
        Note:
        Only the last block of the encoder will be trained.
        """
                
        # Scale 1
        x1_ups = {'streetCornerAtNight':(0,1), 'tramStation':(1,0), 'turbulence2':(1,0)} # ???
        x1 = shared_model.output
        """
        Cropping2D

				keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
				Cropping layer for 2D input (e.g. picture).
				
				It crops along spatial dimensions, i.e. width and height.
				
				Arguments
				
				cropping: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
				If int: the same symmetric cropping is applied to width and height.
				If tuple of 2 ints: interpreted as two different symmetric cropping values for height and width: (symmetric_height_crop, symmetric_width_crop).
				If tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
				data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
				Input shape
				
				4D tensor with shape: - If data_format is "channels_last": (batch, rows, cols, channels) - If data_format is "channels_first": (batch, channels, rows, cols)
				
				Output shape
				
				4D tensor with shape: - If data_format is "channels_last": (batch, cropped_rows, cropped_cols, channels) - If data_format is "channels_first": (batch, channels, cropped_rows, cropped_cols)
				
				Examples
				
				# Crop the input 2D images or feature maps
				model = Sequential()
				model.add(Cropping2D(cropping=((2, 2), (4, 4)),
				                     input_shape=(28, 28, 3)))
				# now model.output_shape == (None, 24, 20, 3)
				model.add(Conv2D(64, (3, 3), padding='same'))
				model.add(Cropping2D(cropping=((2, 2), (2, 2))))
				# now model.output_shape == (None, 20, 16. 64)
        """
        if(self.scene=='wetSnow'):
            x1 = Cropping2D(cropping=((1, 2),(0, 0)))(x1)
        elif(self.scene=='skating'):
            x1 = Cropping2D(cropping=((1, 1),(1, 2)))(x1)
        else:
            for key, val in x1_ups.items():
                if self.scene==key:
                    # upscale by adding number of pixels to each dim.
                    x1 = MyUpSampling2D(size=(1,1), num_pixels=val)(x1)
                    break
        """
        Note:
        	For the first scale, only expending operations are used.
        """
            
        # Scale 2
        x2_ups = {'tunnelExit_0_35fps':(0,1),'tramCrossroad_1fps':(1,0),'bridgeEntry':(1,1),
                  'busyBoulvard':(1,0),'fluidHighway':(0,1),'streetCornerAtNight':(1,1), 
                  'tramStation':(2,0),'winterStreet':(1,0),'twoPositionPTZCam':(1,0),
                  'peopleInShade':(1,1),'turbulence2':(1,1),'turbulence3':(1,0),
                  'skating':(1,1), 'wetSnow':(0,0)}
        
        input_2 = Input(shape=(int(h/2), int(w/2), d), name='ip_scale2')
        x2 = shared_model(input_2)
        x2 = UpSampling2D((2,2))(x2)
        for key, val in x2_ups.items():
            if self.scene == key and self.scene in ['skating', 'wetSnow']:
                x2 = Cropping2D(cropping=((1, 1), val))(x2)
                break
            elif self.scene==key:
                x2 = MyUpSampling2D(size=(1, 1), num_pixels=val)(x2)
                break
        """
        Note:
        	Multiple input fashion in Keras. Nothing but updating and expending.
        """
            
        # Scale 3
        x3_ups = {'tunnelExit_0_35fps':(2,3),'tramCrossroad_1fps':(3,0),'bridgeEntry':(3,1,),
                  'busyBoulvard':(3,0),'fluidHighway':(0,3),'streetCornerAtNight':(1,1),
                  'tramStation':(2,0),'winterStreet':(1,0),'twoPositionPTZCam':(1,2),
                  'peopleInShade':(1,3),'turbulence2':(3,1),'turbulence3':(1,0),
                  'office':(0,2), 'pedestrians':(0,2), 'bungalows':(0,2), 'busStation':(0,2)}
                
        input_3 = Input(shape=(int(h/4), int(w/4), d), name='ip_scale3')
        x3 = shared_model(input_3)
        x3 = UpSampling2D((4,4))(x3)
        for key, val in x3_ups.items():
            if self.scene==key:
                x3 = MyUpSampling2D(size=(1,1), num_pixels=val)(x3)
                break
        """
        Note:
        	Nothing but upsampling and expending.
        """
            
        # concatenate feature maps
        top = keras.layers.concatenate([x1, x2, x3], name='feature_concat')
        if(self.scene=='wetSnow'):
            top = MyUpSampling2D(size=(1,1), num_pixels=(3,0))(top)
        elif(self.scene=='skating'):
            top = MyUpSampling2D(size=(1,1), num_pixels=(2,3))(top)
        """
        Note:
        	Concatenate and expending
        """
        
        # Transposed Conv
        top = self.transposedConv(top)
        # i chose this crazy upscaling/cropping way
        if(self.scene=='tramCrossroad_1fps'):
            top = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(top)
        elif(self.scene=='bridgeEntry'):
            top = MyUpSampling2D(size=(1,1), num_pixels=(2,2))(top)
        elif(self.scene=='fluidHighway'):
            top = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(top)
        elif(self.scene=='streetCornerAtNight'): 
            top = MyUpSampling2D(size=(1,1), num_pixels=(1,0))(top)
            top = Cropping2D(cropping=((0, 0),(0, 1)))(top)
        elif(self.scene=='tramStation'):  
            top = Cropping2D(cropping=((1, 0),(0, 0)))(top)
        elif(self.scene=='twoPositionPTZCam'):
            top = MyUpSampling2D(size=(1,1), num_pixels=(0,2))(top)
        elif(self.scene=='turbulence2'):
            top = Cropping2D(cropping=((1, 0),(0, 0)))(top)
            top = MyUpSampling2D(size=(1,1), num_pixels=(0,1))(top)
        elif(self.scene=='turbulence3'):
            top = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(top)

        vision_model = Model(inputs=[input_1, input_2, input_3], outputs=top, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        vision_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return vision_model
