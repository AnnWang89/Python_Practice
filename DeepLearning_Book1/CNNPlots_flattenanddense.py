# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:10:54 2021

@author: 安ㄢ
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.utils import np_utils

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

model =Sequential()
model.add(Conv2D(filters = 3, kernel_size=(3,3),input_shape=(6,6,1),padding = 'same', name = 'Conv2D_1'))
model.add(Flatten(name ='Flatten_1'))
model.add(Dense(units = 10, activation = 'softmax', name = 'Dense_1'))

SVG(model_to_dot(model, show_shapes = True).create(prog='dot', format='svg'))