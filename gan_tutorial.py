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


##transposed convolution layers --- CONV2DTranspose --
# these are used when we want to transform in the opposite direction of a normal convolution --aka  "DECONVOLUTION"
##also known as "fractionally-strided convolution" 


##Generator takes input  N-dimensional vector 
##GENERATOR will upsample (reshape and upscale) the vector -- into an "image" through deconvolution



class DCGAN: 
    #initial generator starting points -dim=7,depth=64 7x7x64
    def build_discriminator(width, height, depth, alpha=0.2):
        
        dmodel = Sequential()
        #what is alpha ? 
		# initialize the model along with the input shape to be
		# "channels last"
        inputShape = (height, width, depth)
		# first set of CONV => RELU layers
        dmodel.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),input_shape=inputShape))
        dmodel.add(LeakyReLU(alpha=alpha))
		# second set of CONV => RELU layers
        dmodel.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
        dmodel.add(LeakyReLU(alpha=alpha))
		# first (and only) set of FC => RELU layers
        dmodel.add(Flatten())
        dmodel.add(Dense(512))
        dmodel.add(LeakyReLU(alpha=alpha))
		# sigmoid layer outputting a single value
        dmodel.add(Dense(1))
        dmodel.add(Activation("sigmoid"))
		# return the discriminator model
        return dmodel
    
    def build_generator(dim, depth, channels = 1, inputDim= 100, outputDim= 512):
        
        #dim: target spatial dimensions (width and height i.e. 224x224)
        #depth : target depth of volume after reshaping
        
                    # =============================================================================
                    # Depth is the "precision" of each pixel. Typically it can be 8/24/32 bit for 
                    # displaying, but any precision for computations.
                    # 
                    # Instead of precision you can also call it the data type of the pixel. The more 
                    # bits per element, the better to represent different colors or intensities.
                    # 
                    # Your examples mean: 
                    #  8U : 8 bit per element (maybe 8 bit per channel if multiple channels) of unsigned  
                    #       integer type. So probably you can access # elements as unsigned char values, 
                    #       because that's 8 bit unsigned type.
                    # 
                    # 16U : 16 bit per element => unsigned short is typically the 16 bit unsigned integer type on your system.
                    # 
                    # In OpenCV you typically have those types:
                    # 
                    # 8UC3 : 8 bit unsigned and 3 channels => 24 bit per pixel in total.
                    # 
                    # 8UC1 : 8 bit unsigned with a single channel
                    # 
                    # 32S: 32 bit integer type => int
                    # 
                    # 32F: 32 bit floating point => float
                    # 
                    # 64F: 64 bit floating point => double
                    # =============================================================================
            
        #channels: number of channels in output volume - 3 for RGB, 1 for grayscale
        #inputDim: dimensionality of randomly generated input vector
        #outputDim: dimensionality of output FC layer of generator 
        
        
        #so generator takes in some input vector of size 16x16x3 and upscales to 512x512x3
        
        
        model = Sequential()
        
        inputShape = (dim, dim, depth)
        chanDim= -1  ##this sets the channel dimension, assumed to be "channels-last ordering - which is the standard channel ordering for TensorFloW
        #channels-first ordering means image data represetend as channels x rows x cols 
        #channels-last ordering means image data represented as rows x cols x channels
        
        #Build the body of our generator 
        model.add(Dense(input_dim = inputDim, units =outputDim))
        #first FC layer - input size = inputdim =100 and outputsize is 512 
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        #batchnorm addresses the issue of internal covariate shift - this is a phenomenon by which the distribution of data coming out your FC layer is different than what went in. ideally you want mean 0, std_dev = 1 but without batch norm, updating weights of your neurons means that distribution shifts....batchnorm fixes this by normalizing the input values and scaling and shifting it 
        #why would you norm and then ALSO scale and shift? 
        
        
        model.add(Dense(dim*dim*depth))
        #so input is 512 and output is FC layer image volume so 224x224x3 for example
        #above line - prepares number of nodes in FC layer to equal # of units in inputShape; remember inputShape = (dim, dim, depth)
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        
        
        ##actual reshaping 
        model.add(Reshape(inputShape))
        #reshape target size is "inputShape" --why is my inputsize the same as my outputsize..
        model.add(Conv2DTranspose(32, (5,5), strides=(2,2),padding = "same"))
        ##this is the money - 32 filters, each of size 5x5 and appleid with stride(2,2) 
        #what size is our output? 
        #what is stride (2,2)? 
        
        ### transpose/"de-convolution" although it's not technically ...review this
        ### so the size is changing again...getting BIGGER...
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        
       #generator takes an input vector - random noise and upsamples it 
       # to increase spatial resolution --use transposed convolution with stride >1
       # to create deeper GANs without increasing spatial resolution..use standard convolution or transposed convolution but use stride =1 
       
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
        #how do you figure out what the sizes of the output layers should be in practice, when dealing with differently sized inputs?? --pyimage search makes it sound like its going to be a lot of trial and error and debug with model.summary() --not trail and error..use the formula but you will likely stumble at first - that's ok
	   #making sure that # of filters learned is = channels (1-grayscale,3=RGB)
        model.add(Activation("tanh"))
       
        return model
   

