# -*- coding: utf-8 -*-

from keras.models import load_model
from keras import optimizers
from AOD_Net_2 import load_data


model = load_model('aodnet.model')

sgd = optimizers.SGD(lr=0.01, clipvalue=0.1, momentum=0.9, decay=0.001, nesterov=False)
model.compile(optimizer = sgd, loss = 'mean_squared_error')
p_train = 0.8
width = 550
height = 413


data_path = '/home/jianan/Desktop/dongqin_temp/Dataset/test_images_data'
label_path = '/home/jianan/Desktop/dongqin_temp/Dataset/test_images_label'                     
x_train, y_train, x_test, y_test = load_data(data_path, label_path, p_train, height, width)
model.fit(x_train, y_train, epochs = 200, batch_size = 32)
MSE = model.evaluate(x_test, y_test,batch_size = 32)
# use the trained model: model.predict(X_new)
model.save('/home/jianan/Desktop/dongqin_temp/Dataset/aodnet.model')





