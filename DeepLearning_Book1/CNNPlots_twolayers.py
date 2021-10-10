# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:23:42 2021

@author: 安ㄢ
"""

from keras.models import Sequential,Model
from keras.layers import Conv2D
from keras.utils import np_utils

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
os.environ["PATH"] += os.pathsep +'C:/Program Files/Graphviz/bin/' 

model = Sequential()
model.add(Conv2D(filters = 3, kernel_size = (3,3),input_shape = (6,6,1),name = 'Conv2D_1'))
model.add(Conv2D(filters = 2, kernel_size = (3,3),name = 'Conv2D_2'))
SVG(model_to_dot(model, show_shapes = True).create(prog = 'dot',format = 'svg'))