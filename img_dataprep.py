# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 23:06:11 2021

@author: kalan
"""
import imutils
import os 
import cv2
import numpy as np


#resize all images to 128x128x3

#step 1
#resize to width or height to 128. 
#then pad remaining area with zeros such that the final size is 128x128
def resize2SquareKeepingAspectRation(img, size, interpolation):
    
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)


os.chdir(r'C:\Users\kalan\Projects\art_gan\dataset\surrealist art')

###load surreal images
images = []
for filename in os.listdir():
    image =cv2.imread(filename)
    if image is not None:
        images.append(image)

imagesresize =[]
for image in images:
    img= resize2SquareKeepingAspectRation(image, 128, interpolation=cv2.INTER_AREA)
    if img is not None:
        imagesresize.append(img)
    

##load in dali images

os.chdir(r'C:\Users\kalan\Projects\art_gan\dataset\dali')

###load surreal images
images = []
for filename in os.listdir():
    image =cv2.imread(filename)
    if image is not None:
        images.append(image)


for image in images:
    img= resize2SquareKeepingAspectRation(image, 128, interpolation=cv2.INTER_AREA)
    if img is not None:
        imagesresize.append(img)
        
# #remove any duplicates

# if np.any(img2==imagesresize):    
#     print('already there')
# else:     
#     imtest.append(img2)

np.save(r'C:\Users\kalan\Projects\art_gan\dataset\resized.npy', imagesresize) 