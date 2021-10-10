# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:29:00 2021

@author: 安ㄢ
"""

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

model = VGG16(weights='imagenet')

def predict(filename, featuresize):
    img = image.load_img(filename,target_size=(224,224))
    x = image.array_to_img(img)
    x = np.expand_dims(x,axis=0)
    preds =model.predict(preprocess_input(x))
    results = decode_predictions(preds,top = featuresize)[0]
    return results

def showimg(filename,title,i):
    im = Image.open(filename)
    im_list = np.asarray(im)
    plt.subplot(2,5,i)
    plt.title(title)
    plt.axis("off")
    plt.imshow(im_list)
    

filename = "train/cat.3591.jpg"
plt.figure(figsize=(20,10))
for i in range(1):
    showimg(filename,"query", i+1)
plt.show()
results = predict(filename,10)
for result in results:
    print(result)
    
filename = "train/dog.8035.jpg"
plt.figure(figsize=(20,10))
for i in range(1):
    showimg(filename,"query", i+1)
plt.show()
results = predict(filename,10)
for result in results:
    print(result)