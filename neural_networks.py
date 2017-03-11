import numpy as np 
import os
import sys

from keras.optimizers import SGD
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Sequential
from keras.layers import Convolution3D
from keras.layers.convolutional import Deconvolution3D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense
from keras.regularizers import Regularizer
from keras.regularizers import l2, activity_l2
from keras.models import Model
from keras.callbacks import CSVLogger, Callback
from keras import backend as K
from keras.models import load_model
from keras import objectives
from keras.callbacks import ModelCheckpoint
import keras.callbacks
from keras.layers.pooling import MaxPooling3D
from keras.optimizers import RMSprop

class SparseActivityRegularizer(Regularizer):
    def __init__(self, p=0.6, lmbda = 0.6):
        self.p = p
        self.lmbda = lmbda

    def kl_sparse_regularization(self, wgt, rho):
        rho_hat = tf.reduce_mean(wgt)
        invrho = tf.sub(tf.constant(1.), rho)
        invrhohat = tf.sub(tf.constant(1.), rho_hat)
        logrho = tf.add(tf.abs(self.logfunc(rho,rho_hat)), tf.abs(self.logfunc(invrho, invrhohat)))
        return logrho

    def logfunc(self, x1, x2):
        clippDiv = tf.clip_by_value(tf.div(x1,x2),1e-12,1e10)
        return tf.mul( x1,tf.log(clippDiv))

    def __call__(self, layer):
    	logrho = self.kl_sparse_regularization(layer, self.p)
        loss = self.lmbda*logrho
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "p": self.p,
                "lmbda": self.lmbda}


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class ConvolutionalAutoEncoder(object):

	def __init__(self, batch_size, input_size, output_size):
		self.batch_size = batch_size
		self.input_size = input_size
		self.output_size = output_size

	def create_model(self):

		conv_input = Input(batch_shape=tuple(self.input_size), name='main_input')
		num_filters1 = 1
		subsample1 = (1,1,1)
		conv_size1 = 3
		conv_1 = Convolution3D(num_filters1, conv_size1, conv_size1, conv_size1, init='he_normal', border_mode='same',
				 subsample=subsample1,W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				  bias=True, activation=None, name='conv_one')(conv_input)

		conv_bn1 = BatchNormalization(mode=0, axis=4)(conv_1)
		conv_act1 = Activation('relu')(conv_bn1)

		num_filters2 = 1
		subsample2 = (3,3,3)
		conv_size2 = 3
		encode = Convolution3D(num_filters2, conv_size2, conv_size2, conv_size2,init='he_normal', border_mode='same',
				 subsample=subsample2, W_regularizer=None, b_regularizer=None,
				 activity_regularizer=SparseActivityRegularizer(p=0.7, lmbda = 0.6),
				 bias=True, activation=None, name='conv_two')(conv_act1)


		num_defilters1 = 1
		upsample1 = (3,3,3)
		deconv_size1 = 3

		deconv1 = Deconvolution3D(num_defilters1, deconv_size1, deconv_size1, deconv_size1, output_shape=tuple(self.input_size),
				init='he_normal',subsample = upsample1, border_mode='same',
				input_shape = tuple(encode.get_shape().as_list()),activation=None, name='deconv_one')(encode)

		deconv_bn1 = BatchNormalization(mode=0, axis=4)(deconv1)
		deconv_act1 = Activation('relu')(deconv_bn1)

		num_defilters2 = 1
		upsample2 = (1,1,1)
		deconv_size2 = 3
		deconv2 = Deconvolution3D(num_defilters2, deconv_size2, deconv_size2, deconv_size2, output_shape=tuple(self.input_size),
				init='he_normal', subsample = upsample2, border_mode='same', activation=None, name='deconv_two')(deconv_act1)

		convAE = Model(input=conv_input, output=deconv2)
		convAE.compile(optimizer='adam', loss='mean_squared_error')
		convAE.summary()
		self.model = convAE

		return convAE

	def train(self, inTrain, inTest, numEpochs, train_log):
		history = LossHistory()
		csv_logger = CSVLogger(train_log)
		checkpointer = ModelCheckpoint(filepath="final_train2.hdf5", verbose=1, save_best_only=True)

		self.model.fit(inTrain, inTrain,
		        shuffle=True,
		        nb_epoch=numEpochs,
		        batch_size = self.batch_size,
		        validation_data=(inTest, inTest), callbacks=[history, csv_logger, checkpointer])

		self.model.save('conv_autoencoder.hdf5')

	def encode(self, inputData, modelPath):
		print("Loading the model for encoding stage")
		model = self.model
		model.load_weights(modelPath)
		layer_name = 'conv_two'
		encode = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
		encode_output = encode.predict(inputData, batch_size=5, verbose=1)
		return encode_output

	def decode(self, inputData, modelPath):
		print("Loading the model for decoding stage")
		model = self.model
		model.load_weights(modelPath)
		layer_name = 'deconv_two'
		decode = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
		decode_output = encode.predict(inputData, batch_size=5, verbose=1)
		return decode_output



class ConvolutionNeuralNetwork(object):

	def __init__(self, batch_size,output_size):
		self.batch_size = batch_size
		self.output_size = output_size

	def create_model(self):

		K.set_learning_phase(0)
		main_input = Input(batch_shape=(None,31,37,31,1), dtype='float32', name='main_input')

		conv1 = Convolution3D(32, 5, 5, 5, init='he_normal', 
			activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), 
			dim_ordering='tf', W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, 
			W_constraint=None, b_constraint=None, bias=True)(main_input)


		bnorm1 = BatchNormalization(epsilon=0.001, mode=0, axis=4, momentum=0.99, weights=None, 
			beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv1)

		activation1 = Activation('relu')(bnorm1)

		conv2 =Convolution3D(32, 5, 5, 5, init='he_normal',
		 activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), 
		 dim_ordering='tf', W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, 
		 W_constraint=None, b_constraint=None, bias=True)(activation1)


		bnorm2 = BatchNormalization(epsilon=0.001, mode=0, axis=4, momentum=0.99, weights=None, 
			beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv2)

		activation2 = Activation('relu')(bnorm2)

		maxpool1 = MaxPooling3D(pool_size=(5, 5, 5), strides=(1,1,1), 
			border_mode='same', dim_ordering='tf')(activation2)

		conv3 = Convolution3D(32, 5, 5, 5, init='he_normal', 
			activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), dim_ordering='tf', 
			W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, W_constraint=None, 
			b_constraint=None, bias=True)(maxpool1)

		bnorm3 = BatchNormalization(epsilon=0.001, mode=0, axis=4, momentum=0.99, weights=None, 
			beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv3)

		activation3 = Activation('relu')(bnorm3)

		conv4 = Convolution3D(32, 5, 5, 5, init='he_normal', 
			activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), dim_ordering='tf', 
			W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, 
			W_constraint=None, b_constraint=None, bias=True)(activation3)


		bnorm4 = BatchNormalization(epsilon=0.001, mode=0, axis=4, momentum=0.99, weights=None, 
			beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv4)

		activation4 = Activation('relu')(bnorm4)

		maxpool2 = MaxPooling3D(pool_size=(5, 5, 5), strides=(2,2,2), 
			border_mode='same', dim_ordering='tf')(activation4)

		conv5 = Convolution3D(32, 3, 3, 3, init='he_normal', 
			activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), dim_ordering='tf', 
			W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, W_constraint=None, 
			b_constraint=None, bias=True)(maxpool2)

		bnorm5 = BatchNormalization(epsilon=0.001, mode=0, axis=4, momentum=0.99, weights=None, 
			beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv5)

		activation5 = Activation('relu')(bnorm5)

		conv6 =Convolution3D(32, 3, 3, 3, init='he_normal',
		 activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), 
		 dim_ordering='tf', W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, 
		 W_constraint=None, b_constraint=None, bias=True)(activation5)


		bnorm6 = BatchNormalization(epsilon=0.001, mode=0, axis=4, momentum=0.99, weights=None, 
			beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv6)

		activation6 = Activation('relu')(bnorm6)


		maxpool3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1,1,1), 
			border_mode='same', dim_ordering='tf')(activation6)


		conv7 = Convolution3D(16, 3, 3, 3, init='he_normal', 
			activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), dim_ordering='tf', 
			W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, W_constraint=None, 
			b_constraint=None, bias=True)(maxpool3)

		bnorm7 = BatchNormalization(epsilon=0.001, mode=0, axis=4, momentum=0.99, weights=None, 
			beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv7)

		activation7 = Activation('relu')(bnorm7)

		conv8 = Convolution3D(1, 3, 3, 3, init='he_normal', 
			activation=None, weights=None, border_mode='same', subsample=(1, 1, 1), dim_ordering='tf', 
			W_regularizer=l2(0), b_regularizer=l2(0), activity_regularizer=None, 
			W_constraint=None, b_constraint=None, bias=True)(activation7)


		bnorm8 = BatchNormalization( mode=0, axis=4, momentum=0.99,
			beta_init='zero',gamma_init='one')(conv8)

		activation8 = Activation('relu')(bnorm8)

		maxpool4 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2,2,2), 
			border_mode='same', dim_ordering='tf')(activation8)

		# maxpool_tuple=(maxpool4.get_shape())

		# output_dim1 = maxpool_tuple[1]*maxpool_tuple[2]*maxpool_tuple[3]*maxpool_tuple[4]


		flattened_matrix = Flatten()(maxpool4)
		# flattened_matrix = K.reshape(maxpool4, [-1, np.prod(maxpool4.get_shape()[1:].as_list())])


		nnet1=Dense(32, init='he_normal', activation=None, weights=None, W_regularizer=l2(0), 
			b_regularizer=l2(0), activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)(flattened_matrix)


		# bnorm9 = BatchNormalization(epsilon=0.001, mode=0, axis=0, momentum=0.99, weights=None, 
		# 	beta_init='zero',gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(nnet1)

		activation9 = Activation('relu')(nnet1)


		output_dim2 = self.output_size
		nnet2=Dense(output_dim2, init='he_normal', activation=None, weights=None, W_regularizer=l2(0), 
			b_regularizer=l2(0), activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)(activation9)

		activation10=Activation('sigmoid')(nnet2)

		convNetwork = Model(input=main_input, output=activation10)

		
		convNetwork.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		convNetwork.summary()
		self.model=convNetwork

	def train(self, X_train, Y_train, X_val, Y_val, numEpochs,train_log):

		history = LossHistory()
		csv_logger = CSVLogger(train_log)
		checkpointer = ModelCheckpoint(filepath="cnn_checkpoint.hdf5", verbose=1, save_best_only=True)

		
		hist = self.model.fit(X_train, Y_train, # Train the model using the training set...
          batch_size=self.batch_size, nb_epoch=numEpochs,shuffle='batch',
          verbose=1,			
          callbacks=[csv_logger, checkpointer],validation_data=(X_val, Y_val)) 

		print(hist.history)

		self.model.save('cnn_model.hdf5')

	def test(self, X_test,modelPath):
		print("Loading the model for testing and prediction purposes")
		model = self.model
		model.load_weights(modelPath)
		probabilites = model.predict(X_test, batch_size=32, verbose=1)
		print('Probabilities are : '+str(probabilites))
		pred = np.around(probabilites)
		print('Probabilities after rounding off are : '+str(pred))
		return pred



















