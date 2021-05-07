# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 10:41:17 2021

@author: kalan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:02:12 2021

@author: kalan
"""

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape


class DCGAN: 
    #initial generator starting points -dim=7,depth=64 7x7x64
    def build_discriminator(width, height, depth, alpha=0.2):
        
        model = Sequential()
        #what is alpha ? 
		# initialize the model along with the input shape to be
		# "channels last"
        inputShape = (height, width, depth)
		# first set of CONV => RELU layers
        model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))
		# second set of CONV => RELU layers
        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
        model.add(LeakyReLU(alpha=alpha))
		# first (and only) set of FC => RELU layers
        # model.add(Conv2D(128, (5, 5), padding="same", strides=(2, 2)))
        # model.add(LeakyReLU(alpha=alpha))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))
		# sigmoid layer outputting a single value
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
		# return the discriminator model
        return model
    
    def build_generator(dim, depth, channels = 3, inputDim= 100, outputDim= 512):
        
        
        model = Sequential()
        
        inputShape = (dim, dim, depth)
        chanDim= -1  ##this sets the channel dimension, assumed to be "channels-last ordering - 
        
        #Build the body of our generator 
        model.add(Dense(input_dim = inputDim, units =outputDim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        
        #batchnorm addresses the issue of internal covariate shift
        model.add(Dense(dim*dim*depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        
        
        ##actual reshaping 
        model.add(Reshape(inputShape))


        model.add(Conv2DTranspose(128, (5,5), strides=(2,2),padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2DTranspose(64, (5,5), strides=(2,2),padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2DTranspose(32, (5,5), strides=(2,2),padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        # model.add(Conv2DTranspose(16, (5,5), strides=(2,2),padding = "same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        
        
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("tanh"))
       
        return model
   

