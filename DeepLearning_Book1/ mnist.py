# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 13:29:37 2021

@author: 安ㄢ
"""

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger

import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 20

(x_train,y_train), (x_test,y_test) = mnist.load_data()

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.title("M_%d"%i)
    plt.axis("off")
    plt.imshow(x_train[i].reshape(28,28),cmap = None)
    
plt.show()

x_train = x_train.reshape(60000,784).astype('float32')
x_test = x_test.reshape(10000,784).astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(512, input_shape = (784, )))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(),metrics = ['accuracy'])

es = EarlyStopping(monitor = 'val_loss', patience = 2)
csv_logger = CSVLogger('training.log')
hist = model.fit(x_train,y_train,batch_size=batch_size,epochs= epochs,verbose=1,validation_split=0.1,callbacks=[es, csv_logger])

score = model.evaluate(x_test,y_test,verbose=0)
print('test loss:',score[0])
print('test acc:',score[1])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs),loss,marker='.',label = 'loss(training data')
plt.plot(range(epochs),val_loss,marker='.',label = 'val_loss(evaluation data')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
