# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:50:19 2021

@author: kalan
"""

# ======================================================================
#         
#In their paper, Radford et al. recommend the following architecture guidelines for more stable GANs:
# 
# Replace any pooling layers with strided convolutions 
# Use batch normalization in both the generator and discriminator.
# Remove fully-connected layers in deeper networks.
# Use ReLU in the generator except for the final layer, which will utilize tanh.
# Use Leaky ReLU in the discriminator.
# ======================================================================





 =============================================================================
# In his book, Francois Chollet then provides additional recommendations on training GANs:
# 
# Sample random vectors from a normal distribution (i.e., Gaussian distribution) rather than a uniform distribution.
# Add dropout to the discriminator.
# Add noise to the class labels when training the discriminator.
# To reduce checkerboard pixel artifacts in the output image, use a kernel size that is divisible by the stride when utilizing convolution or transposed convolution in both the generator and discriminator.
# If your adversarial loss rises dramatically while your discriminator loss falls to zero, try reducing the learning rate of the discriminator and increasing the dropout of the discriminator.
# =============================================================================

from imutils import build_montages
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from gan_tutorial import DCGAN

NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
INIT_LR = 2e-4



##download the data from fashion_mnist and then stack the train and test into one big array
((trainX,_), (testX, _)) = fashion_mnist.load_data()
trainImages = np.concatenate([trainX,testX])

#add an extra dimension for the channel and scale the images into range [-1,1] (range of tanh function)
trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") -127.5)/127.5



generator = DCGAN.build_generator(7,64,channels=1)
generator.summary()

discriminator = DCGAN.build_discriminator(28,28,1)
discriminator.summary()



##learning rate and beta value for Adam optimizer experimentally tuned -- you will need to run tons of experiments
d_opt = Adam(lr=INIT_LR, beta_1 =0.5, decay = INIT_LR / NUM_EPOCHS)
discdiscriminatorcompile(loss="binary_crossentropy",optimizer=d_opt)


#build the adversarial model - set the discrimintor to NOT be trainable
discriminator.trainable = False
ganInput = Input(shape=(100,))
ganOutput = discriminator(generator(ganInput))

#combine the generator and discriminator to for m the GAN
gan = Model(ganInput,ganOutput)
#compile the GNA
gan.compile(loss ="binary_crossentropy", optimizer=d_opt)


#randomly generate benchmark noise - to visualize how generative modeling is learning
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))

#for every poch
for epoch in range(0, NUM_EPOCHS):
    #show epoch # of total NUM EPOCHS and track progress
    print("[INFO] starting epoch {} of {}...".format(epoch+1, NUM_EPOCHS))
    batchesPerEpoch = int(trainImages.shape[0]/ BATCH_SIZE)
    
    #loop over batches
    for i in range(0, batchesPerEpoch):
        #initalize empty output p
        p=None
        
        imageBatch = trainImages[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
        noise= np.random.uniform(-1,1, size = (BATCH_SIZE,100))
        
        #generate images using noise input
        genImages = generator.predict(noise, verbose =0)
        
        #concatenate actual images and generated images
        X = np.concatenate((imageBatch, genImages))
        
        y = ([1]*BATCH_SIZE) + ([0]*BATCH_SIZE)
        y = np.reshape(y,(-1,))
        (X,y) = shuffle(X,y)
        
        
        
        #train disciriminator 
        discLoss = discriminator.train_on_batch(X,y)
