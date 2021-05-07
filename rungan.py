# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:46:34 2021

@author: kalan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:50:19 2021

@author: kalan
"""

from imutils import build_montages
#from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
##cwd --> C:\Users\kalan\Projects\art_gan
from surrealgan import DCGAN

#os.chdir(r'C:\Users\kalan\Projects\art_gan')

ap=argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-e","--epochs", type = int, default = 3 , help="#epochs to train for")
ap.add_argument("-b","--batch-size", type = int, default = 16 , help="batch size for training samples")
args = vars(ap.parse_args())

NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
INIT_LR = 2e-4



print("[INFO] Loading Surreal art data...")
##load new data
trainImages = np.load('dataset/resized.npy')
trainImages = (trainImages.astype("float") -127.5)/127.5



print("[INFO] building generator...")
generator = DCGAN.build_generator(8,64,channels=3)
#enerator.summary()
print("[INFO] building discriminator...")
discriminator = DCGAN.build_discriminator(128,128,3)
#discriminator.summary()



##learning rate and beta value for Adam optimizer experimentally tuned -- you will need to run tons of experiments
d_opt = Adam(lr=INIT_LR, beta_1 =0.5, decay = INIT_LR / NUM_EPOCHS)
discriminator.compile(loss="binary_crossentropy",optimizer=d_opt)


#build the adversarial model - set the discrimintor to NOT be trainable
discriminator.trainable = False

ganInput = Input(shape=(100,))
#why do i need this?
ganOutput = discriminator(generator(ganInput))

#combine the generator and discriminator to for m the GAN
gan = Model(ganInput,ganOutput)
#compile the GNA
gan.compile(loss ="binary_crossentropy", optimizer=d_opt)

print("[INFO] beginning training...")
#randomly generate benchmark noise - to visualize how generative modeling is learning
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))
#generated from uniform distributionwith range [-1,1]--like tanh fn
#generates 256 noise vectors - each a 100d vector
#why? ..

#for every poch
for epoch in range(0, NUM_EPOCHS):
    #show epoch # of total NUM EPOCHS and track progress
    print("[INFO] starting epoch {} of {}...".format(epoch+1, NUM_EPOCHS))
    batchesPerEpoch = int(trainImages.shape[0]/ BATCH_SIZE)
    
    #loop over batches

    for i in range(0, batchesPerEpoch):
    
        #initalize empty output p
        p=None
        
        #move through the trainImages one batch size set of images at a time
        imageBatch = trainImages[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
        #generate another noise vector to pass through to generator -- why again?
        noise= np.random.uniform(-1,1, size = (BATCH_SIZE,100))
        
        #generate images using noise input
        genImages = generator.predict(noise, verbose =0)
        
        #combine actual images and generated images
        X = np.concatenate((imageBatch, genImages))
        
        #construct class labels for the discriminator 128 1s and 128 0s 
        y = ([1]*BATCH_SIZE) + ([0]*BATCH_SIZE)
        #convert from list to np array 
        y = np.reshape(y,(-1,))
        
        #shuffle data up  - but at this stage we've identified real images and fake images for the discriminator
        (X,y) = shuffle(X,y)
        
        
        
        #train disciriminator 
        discLoss = discriminator.train_on_batch(X,y)



    #final step - after generating images, and sending to discriminator to train --train generator via adversarial model
        #generate another noise dataset
        noise= np.random.uniform(-1,1,(BATCH_SIZE,100))
        
        fakeLabels = [1] * BATCH_SIZE
        fakeLabels = np.reshape(fakeLabels, (-1,))
        #trying to trick dsicimrinator by sending noise with fakelabels??
        ganLoss = gan.train_on_batch(noise, fakeLabels)
        
        # check to see if this is the end of an epoch, and if so,
		# initialize the output path
        if i == batchesPerEpoch - 1:
            p = [args["output"], "epoch_{}_output.png".format(
				str(epoch + 1).zfill(4))]
		# otherwise, check to see if we should visualize the current
		# batch for the epoch
        else:
			# create more visualizations early in the training
			# process
            if epoch < 10 and i % 25 == 0:
                p = [args["output"], "epoch_{}_step_{}.png".format(
					str(epoch + 1).zfill(4), str(i).zfill(5))]
			# visualizations later in the training process are less
			# interesting
            elif epoch >= 10 and i % 100 == 0:
                p = [args["output"], "epoch_{}_step_{}.png".format(
					str(epoch + 1).zfill(4), str(i).zfill(5))]
                
                
        if p is not None:
        
		# show loss information
            print("[INFO] Step {}_{}: discriminator_loss={:.6f}, "
			"adversarial_loss={:.6f}".format(epoch + 1, i,
				discLoss, ganLoss))
		# make predictions on the benchmark noise, scale it back
		# to the range [0, 255], and generate the montage
            images = generator.predict(benchmarkNoise)
            images = ((images * 127.5) + 127.5).astype("uint8")
            #images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (128, 128), (4, 4))[0]
            #write the the visualization to disk
            p = os.path.sep.join(p)
            #print("[INFO] Writing to output...")
            cv2.imwrite(p, vis)

