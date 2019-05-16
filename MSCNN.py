import cv2
import numpy as np
import random
import os
import keras.backend as K

from keras.layers import Conv2D, Input, UpSampling2D, concatenate, MaxPooling2D
from keras import optimizers
from keras.models import Model
from keras.activations import sigmoid
from keras.engine.topology import Layer
from keras.callbacks import LearningRateScheduler

def load_data(data_files,label_files, height, width):
    
    data = []
    label = []
    
    for data_file in data_files:
        hazy_image = cv2.imread(data_path + "/" + data_file)
        label_file = label_files[label_files.index(data_file.partition('.')[0][:-2] + data_file[-4:])]
        trans_map = cv2.imread(label_path + "/" + label_file, 0)
        
        if hazy_image.shape != (height, width, 3):
            hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
            trans_map = cv2.resize(trans_map, (width, height), interpolation = cv2.INTER_AREA)
            
        data.append(hazy_image)
        label.append(trans_map)
    
    data = np.asarray(data) / 255.0 # whether to normalize
    label = np.asarray(label).reshape(len(label), height, width, 1) / 255.0
    
    return data, label

def get_batch(data_files, label_files, batch_size, height, width):
   
    while 1:
        for i in range(0, len(data_files), batch_size):
            x, y = load_data(data_files[i : i+batch_size], label_files, height, width)
            
            yield x, y
            
class Linear_Comb(Layer):
    '''
    https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/
    https://blog.csdn.net/u013084616/article/details/79295857
    http://kibo.tech/2018/08/12/56-%E8%AE%A9Keras%E6%9B%B4%E9%85%B7%E4%B8%80%E4%BA%9B%EF%BC%81Keras%E6%A8%A1%E5%9E%8B%E6%9D%82%E8%B0%88/
    '''
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Linear_Comb, self).__init__(**kwargs)
         
    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', 
                                      shape = (input_shape[3],self.output_dim), 
                                      initializer='random_normal', 
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.output_dim,),
                                    initializer='random_normal', 
                                    trainable=True)
        super(Linear_Comb, self).build(input_shape)
    
    def call(self, x):
        return sigmoid(K.dot(x, self.kernel) + self.bias)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)
    
def get_airlight(hazy_image, trans_map, p):
    M, N = trans_map.shape
    flat_image = hazy_image.reshape(M*N, 3)
    flat_trans = trans_map.ravel()
    searchidx = (-flat_trans).argsort()[:round(M * N * p)]
    
    return np.max(flat_image.take(searchidx, axis=0), axis = 0)

def get_radiance(hazy_image, airlight, trans_map, L):
    
    tiledt = np.zeros_like(hazy_image)
    tiledt[:,:,0] = tiledt[:,:,1] = tiledt[:,:,2] = trans_map
    min_t = np.ones_like(hazy_image) * 0.1
    t = np.maximum(tiledt, min_t)
    
    hazy_image = hazy_image.astype(int)
    airlight = airlight.astype(int)
    
    clear_ = (hazy_image - airlight) / t + airlight
    clear_image = np.maximum(np.minimum(clear_, L-1), 0).astype(np.uint8)
    
    return clear_image
    
def MSCNN():
    input_image = Input(shape = (None, None, 3), name = 'c_input')
    c_conv1 = Conv2D(5, (11,11), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'c_conv1')(input_image)
    c_mp1 = MaxPooling2D(pool_size = (2,2), padding = 'valid', name = 'c_mp1')(c_conv1)
    c_up1 = UpSampling2D(size=(2,2), interpolation = 'nearest', name = 'c_up1')(c_mp1)
    c_conv2 = Conv2D(5, (9,9), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'c_conv2')(c_up1)
    c_mp2 = MaxPooling2D(pool_size = (2,2), padding = 'valid', name = 'c_mp2')(c_conv2)
    c_up2 = UpSampling2D(size=(2,2), interpolation = 'nearest', name = 'c_up2')(c_mp2)
    c_conv3 = Conv2D(10, (7,7), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'c_conv3')(c_up2)
    c_mp3 = MaxPooling2D(pool_size = (2,2), padding = 'valid', name = 'c_mp3')(c_conv3)
    c_up3 = UpSampling2D(size=(2,2), interpolation = 'nearest', name = 'c_up3')(c_mp3)
    ### if self defined layer does not work, consider using this convolutional layer
    #c_linear = Conv2D(1, (1,1), strides=(1,1), padding ='same', activation='sigmoid',kernel_initializer='random_normal', name = 'c_linear')(c_up3)
    ###
    c_linear = Linear_Comb(1, name = 'c_linear')(c_up3)
   
    f_conv1 = Conv2D(4, (7,7), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'f_conv1')(input_image)
    f_mp1 = MaxPooling2D(pool_size = (2,2), padding = 'valid', name = 'f_mp1')(f_conv1)
    f_up1 = UpSampling2D(size=(2,2), interpolation = 'nearest', name = 'f_up1')(f_mp1)
    concat1 = concatenate([f_up1, c_linear], axis = -1, name = 'f_concat')
    f_conv2 = Conv2D(5, (5,5), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'f_conv2')(concat1)
    f_mp2 = MaxPooling2D(pool_size = (2,2), padding = 'valid', name = 'f_mp2')(f_conv2)
    f_up2 = UpSampling2D(size=(2,2), interpolation = 'nearest', name = 'f_up2')(f_mp2)
    f_conv3 = Conv2D(10, (3,3), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'f_conv3')(f_up2)
    f_mp3 = MaxPooling2D(pool_size = (2,2), padding = 'valid', name = 'f_mp3')(f_conv3)
    f_up3 = UpSampling2D(size=(2,2), interpolation = 'nearest', name = 'f_up3')(f_mp3)
    ### if self defined layer does not work, consider using this convolutional layer
    #f_linear = Conv2D(1, (1,1), strides=(1,1), padding ='same', activation='sigmoid',kernel_initializer='random_normal', name = 'f_linear')(f_up3)
    ###
    f_linear = Linear_Comb(1, name= 'f_linear')(f_up3)
    model = Model(inputs = input_image, outputs = f_linear)
    return model

def train_model(data_path, label_path, weights_path, lr=0.001, momentum=0.9, decay=5e-4, p_train = 0.8, 
                width = 320, height = 240, batch_size = 100, nb_epochs = 40):
    
    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(sgd.lr)
            K.set_value(sgd.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(sgd.lr)
    
    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
    
    p_train = p_train
    width = width
    height = height
    batch_size = batch_size
    nb_epochs = nb_epochs
    
    data_files = os.listdir(data_path) # seems os reads files in an arbitrary order
    label_files = os.listdir(label_path)
    
    random.seed(100)  # ensure we have the same shuffled data every time
    random.shuffle(data_files) 
    data_files = data_files[0:30000]
    x_train = data_files[0: round(len(data_files) * p_train)]
    x_val =  data_files[round(len(data_files) * p_train) : len(data_files)]
    if len(x_train) % batch_size == 0:
        steps_per_epoch = len(x_train) // batch_size
    else:
        steps_per_epoch = len(x_train) // batch_size + 1
        
    if len(x_val) % batch_size == 0:
        steps = len(x_val) // batch_size
    else:
        steps = len(x_val) // batch_size + 1
    
    reduce_lr = LearningRateScheduler(scheduler)
    
    mscnn = MSCNN()
    mscnn.summary()
    mscnn.compile(optimizer = sgd, loss = 'mean_squared_error')
    mscnn.fit_generator(generator = get_batch(x_train, label_files, batch_size, height, width), 
                        steps_per_epoch=steps_per_epoch, epochs = nb_epochs, validation_data = 
                        get_batch(x_val, label_files, batch_size, height, width), validation_steps = steps,
                        use_multiprocessing=True, 
                        shuffle=False, initial_epoch=0, callbacks = [reduce_lr])
    mscnn.save_weights(weights_path + '/mscnn.h5')
    print('MSCNN generated')
    
    return weights_path + '/mscnn.h5'

def Load_model(weights):
    mscnn = MSCNN()
    mscnn.load_weights(weights)
    
    return mscnn

def usemodel(mscnn, hazy_image):
    
    height = hazy_image.shape[0]
    width = hazy_image.shape[1]
    channel = hazy_image.shape[2]
    
    p = 0.001
    L = 256
        
    if height % 2 != 0:
        height = hazy_image.shape[0] // 2 * 2
    if width % 2 != 0:
        width = hazy_image.shape[1] // 2 * 2
    
    hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
    hazy_input = np.reshape(hazy_image, (1, height, width, channel))
    trans_map = mscnn.predict(hazy_input)
    trans_map = np.reshape(trans_map, (height, width))
    Airlight = get_airlight(hazy_image, trans_map, p)
    clear_image = get_radiance(hazy_image, Airlight, trans_map, L)
    
    return clear_image

if __name__ =="__main__":
    '''
    Possible modification:
        code MSCNN as a single network -- this one could be most helpful
        normalize the images
        set batch_size to 100
        nb_epochs = 70
        change lr every 20 epochs
    '''
    data_path = ''
    label_path = ''
    weights_path = ''
    mscnn_weights = train_model(data_path, label_path, weights_path)
    
    im_path = ''
    mscnn = Load_model(mscnn_weights)
    
    im = cv2.imread(im_path)
    im_dehaze = usemodel(mscnn, im)
