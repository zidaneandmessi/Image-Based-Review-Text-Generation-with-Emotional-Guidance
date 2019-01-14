import time
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np

t0 = time.time()
height,width = 800,800
img_path = 'data/test.jpg'
img = image.load_img(img_path,target_size=(800,800))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)


print("开始建模CNN ...")
model = Sequential()
# Block 1, 2层
model.add(Convolution2D(64, 4, 4, activation='relu',
                    border_mode='same', input_shape=(3,height, width)))
model.add(Convolution2D(64, 4, 4, activation='relu',border_mode='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
 
# Block 2, 2层
model.add(Convolution2D(128, 4, 4, activation='relu',border_mode='same'))
model.add(Convolution2D(128, 4, 4, activation='relu',border_mode='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
 
# Block 3, 3层
model.add(Convolution2D(256, 4, 4, activation='relu',border_mode='same'))
model.add(Convolution2D(256, 4, 4, activation='relu',border_mode='same'))
model.add(Convolution2D(256, 4, 4, activation='relu',border_mode='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
 
# Classification block, 全连接3层
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))
model.load_weights('vgg16_weights_th_dim_ordering_th_kernels.h5')
print("建模CNN完成 ...")


y_pred = model.predict(x)
print(img_path,decode_predictions(y_pred))
print(decode_predictions(y_pred)[0][0][1])


