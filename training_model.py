import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU') #gpus[0], gpus[1], gpus[2]
##########################################################################################################
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input,Conv2D, Conv1D, ELU, MaxPooling2D, Flatten, Dense, Dropout, Reshape, DepthwiseConv2D, Lambda, LSTM
from tensorflow.keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import scipy.io as sio
import time

def get_lstm_layer(input_layer):
	
	lstm_output = LSTM(units = 5, return_sequences = True)(input_layer)
	
	return lstm_output

def get_atrous_layers(input_layer):
	# AtrousConv Layers
	# apply an atrous convolution 1d
	# with atrous rates 6, 12, 18, 24 of length 3 to a sequence with 1024 feature maps,
	# with 64 output filters
	atr_conv1 = Conv1D(64, 3, dilation_rate=6, padding='same')(input_layer)
	atr_2dconv1 = Conv1D(32, 1, padding='same')(atr_conv1)
	atr_elu1 = ELU()(atr_2dconv1)
	
	atr_conv2 = Conv1D(64, 3, dilation_rate=12, padding='same')(input_layer)
	atr_2dconv2 = Conv1D(32, 1, padding='same')(atr_conv2)
	atr_elu2 = ELU()(atr_2dconv2)
	
	atr_conv3 = Conv1D(64, 3, dilation_rate=18, padding='same')(input_layer)
	atr_2dconv3 = Conv1D(32, 1, padding='same')(atr_conv3)
	atr_elu3 = ELU()(atr_2dconv3)
	
	return atr_elu1, atr_elu2, atr_elu3
#%%###
def get_model():
	stream1=Input(shape=(81, 512, 3))
	#conv1
	stream1=Conv2D(32, (3, 3), padding='same', name='conv1')(stream1)
	stream1=ELU()(stream1)
	stream1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(stream1)
	#conv2
	stream1=Conv2D(32, (3, 3), padding='same', name='conv2')(stream1)
	stream1=ELU()(stream1)
	stream1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(stream1)
	#conv3
	stream1=Conv2D(64, (3, 3), padding='same', name='conv3')(stream1)
	stream1=ELU()(stream1)
	#conv4
	stream1=Conv2D(64, (3, 3), padding='same', name='conv4')(stream1)
	stream1=ELU()(stream1)
	#conv5
	stream1=Conv2D(128, (3, 3), padding='same', name='conv5')(stream1)
	stream1=ELU()(stream1)
	stream1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5')(stream1)
	#Reshape
	stream1 = Reshape((-1, 128))(stream1)
	###########################################
	lstm1 = get_lstm_layer(stream1)
	###########################################
	atr_output1, atr_output2, atr_output3 = get_atrous_layers(lstm1)
	
	###########################################
	#right image
	stream2 = Input(shape=(81, 512, 3))
	#conv1
	stream2=Conv2D(32, (3, 3), padding='same', name='conv6')(stream2)
	stream2=ELU()(stream2)
	stream2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool6')(stream2)
	#conv2
	stream2=Conv2D(32, (3, 3), padding='same', name='conv7')(stream2)
	stream2=ELU()(stream2)
	stream2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool7')(stream2)
	#conv3
	stream2=Conv2D(64, (3, 3), padding='same', name='conv8')(stream2)
	stream2=ELU()(stream2)
	#conv4
	stream2=Conv2D(64, (3, 3), padding='same', name='conv9')(stream2)
	stream2=ELU()(stream2)
	#conv5
	stream2=Conv2D(128, (3, 3), padding='same', name='conv10')(stream2)
	stream2=ELU()(stream2)
	stream2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool10')(stream2)
	#Reshape
	stream2 = Reshape((-1, 128))(stream2)
	
	#############################################
	lstm2 = get_lstm_layer(stream2)
	
	#############################################
	atr_output4, atr_output5, atr_output6 = get_atrous_layers(lstm2)
	
	############################################
	#concatenate layerss
	atr_fusion1 = keras.layers.add([atr_output1, atr_output2, atr_output3])
	atr_fusion2 = keras.layers.add([atr_output4, atr_output5, atr_output6])
	
	############################################
	fusion_lstm1 = get_lstm_layer(atr_fusion1)
	fusion_lstm2 = get_lstm_layer(atr_fusion2)
	
	############################################
	
	fusion3_drop7 = keras.layers.concatenate([fusion_lstm1, fusion_lstm2])
	##############################################
	#fc6
	flat6 = Flatten()(fusion3_drop7)
	fc6 = Dense(400)(flat6)
	elu6 = ELU()(fc6)
	drop6 = Dropout(0.35)(elu6)
	
	fc66 = Dense(200)(drop6)
	elu66 = ELU()(fc66)
	drop66 = Dropout(0.35)(elu66)
	
	#fc8
	fusion3_fc8=Dense(100)(drop66)
	#fc9
	predictions=Dense(1)(fusion3_fc8)
	
	model_all=Model([left_image, right_image], predictions, name='all_model')
	model_all.summary()
	
	return model_all
	###############################################
#%% train model
def compile_model(model):
	sgd=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1) #lr=0.0001
	model.compile(loss='mean_squared_error', optimizer=sgd)
	
	return model

def run_model(model, stream1_input, stream2_input, labels):
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
	mc = ModelCheckpoint('model/2stream_model.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

	#fitting model
	history = model.fit(x=[stream1_input, stream2_input], y=[labels], validation_split=0.2, batch_size=128, epochs=6000, verbose=1, callbacks=[es, mc], shuffle=True)
	
	#saving history
	np.save('2stream_history.npy',history.history)
