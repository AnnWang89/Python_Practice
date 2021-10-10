# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:32:40 2021

@author: 安ㄢ
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import glob,cv2

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()
    
def show_images_labels_predictions(images,labels,predictions,start_id,num=10):
    plt.gcf().set_size_inches(12,14)
    if num > 25:
        num = 25
    for i in range(num):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(images[start_id], cmap='binary')
        
        if (len(predictions) > 0):
            title = 'ai = ' + str(predictions[start_id])
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)')
            title += '\nlabel = ' + str(labels[start_id]) 
        else:
            title = 'labels = ' + str(labels[start_id])
            
        ax.set_title(title,fontsize = 12 )
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1
    plt.show()
    
(train_feature, train_label),(test_feature, test_label) = mnist.load_data()
test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')

test_feature_normalize = test_feature_vector/255

print('load model Mnist_mlp_model.h5')
model = load_model('Mnist_mlp_model.h5')
prediction = model.predict_classes(test_feature_normalize)
show_images_labels_predictions(test_feature,test_label,prediction,0)

files = glob.glob('Minst_data\*.png')
test_feature=[]
test_label =[]
for file in files:
    img=cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
    test_feature.append(img)
    label = file[11:12]
    test_label.append(int(label))
    
test_feature = np.array(test_feature)
test_label = np.array(test_label)
test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')
test_feature_normalize = test_feature_vector/255
prediction = model.predict_classes(test_feature_normalize)
show_images_labels_predictions(test_feature,test_label,prediction,0,len(test_feature))