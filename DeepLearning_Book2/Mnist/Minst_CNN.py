# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:10:03 2021

@author: 安ㄢ
"""

from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import load_model
import matplotlib.pyplot as plt
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
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

train_feature_vector = train_feature.reshape(len(train_feature),28,28,1).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature),28,28,1).astype('float32')

train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

train_label_onehot = np_utils.to_categorical(train_label) 
test_label_onehot = np_utils.to_categorical(test_label)

model = Sequential()
model.add(Conv2D(filters = 10,
                 kernel_size = (3,3),
                 padding = 'same',
                 input_shape = (28,28,1),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 20,
                 kernel_size = (3,3),
                 padding = 'same',
                 input_shape = (28,28,1),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=256, activation = 'relu'))
model.add(Dense(units=10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics = ['accuracy'])
train_history =model.fit(x = train_feature_normalize,
                         y = train_label_onehot, validation_split=0.2,
                         epochs = 10, batch_size = 200, verbose = 2)

scores = model.evaluate(test_feature_normalize, test_label_onehot)
print('\n準確率  = ' ,scores[1])
model.save('Mnist_cnn_model.h5')
print('\nMnist_cnn_model.h5 儲存完畢')
model.save_weights("Mnist_cnn_model.weight")
print('\nMnist_cnn_model.weight 儲存完畢')
del model
model = load_model('Mnist_cnn_model.h5')

prediction = model.predict_classes(test_feature_normalize)
show_images_labels_predictions(test_feature, test_label,prediction,0,len(test_feature))