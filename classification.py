import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

#variables

img_rows = 64
img_cols = 64
num_channel = 1

sketch_data_list = []	

# Reading all sketches from the directory

path = "C:/Users/chandan/Desktop/Final_year_project/dataset/data"
data_list = os.listdir(path)
#print(data_list)
for dataset in data_list:
	#print(dataset)
	dataset_path = os.path.join(path, dataset)
	#print(dataset_path)
	sketch_list = os.listdir(dataset_path)
	for sketch in sketch_list :
		#print(sketch)
		sketch_path = os.path.join(dataset_path, sketch)
		input_sketch = cv2.imread(sketch_path)
		input_sketch = cv2.cvtColor(input_sketch, cv2.COLOR_BGR2GRAY)
		input_sketch_rf = cv2.resize(input_sketch, (img_rows, img_cols))
sketch_data = np.array(sketch_data_list)
sketch_data = sketch_data.astype('float32')
sketch_data /= 255
print(sketch_data.shape) 


#sketch_data_normalized = preprocessing.normalize(sketch_data) #Scale/Normlaize#
'''print(sketch_data_normalized.shape)
print(np.mean(sketch_data_normalized))
print(np.std(sketch_data_normalized))'''

# dimensional ordering
sketch_data = np.expand_dims(sketch_data, axis = 4)
print(sketch_data.shape)