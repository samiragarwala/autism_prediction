# Main file for running the programs
import numpy as np 
import keras
import sys
import os
import h5py
import random
import matplotlib as ml
ml.use("agg")
import matplotlib.pyplot as plt
from neural_networks import ConvolutionalAutoEncoder, ConvolutionNeuralNetwork
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.utils.io_utils import HDF5Matrix
from utils import *

def preprocessDatasets(hdf5File):
	hf = h5py.File(fileInputs,'r')
	datasets = hf.keys()
	Xtot = np.zeros((len(datasets), 91, 109, 91, 1))
	count = 0
	for key in datasets:
		curData = hf.get(key)
		Xtot[count,:,:,:,0] = np.array(curData)
		count += 1

	for i in range(0,count):
		mu = np.mean(Xtot[i,:,:,:,0])
		var = np.std(Xtot[i,:,:,:,0])
		Xtot[i,:,:,:,0] = (Xtot[i,:,:,:,0] - mu)/var

	return Xtot

def convAutoEncoderTrain():
	fileInputs = sys.argv[1]
	print("Reading HDF5 File")
	
	Xtot = preprocessData(fileInputs)
	print("Creating Convolutional Autoencoder Object")
	batch_size = 32
	inputSize = Xtot[0,:,:,:,:].shape
	cae = ConvolutionalAutoEncoder(32,(None,) + inputSize, (None,) + inputSize)

	print("Defining the Convolutional AutoEncoder model")
	convAeModel = cae.create_model()

	print("Training the Convolutional AutoEncoder")
	numEpochs = 60

	cae.train(Xtot[:920,:,:,:,:], Xtot[920:,:,:,:,:], numEpochs, 'convAutoEncoder.log')

	# mat2visual(Xtot[2,:,:,:,0], [40,45,60], 'original_train.png', )
	# mat2visual(encode_output_train[2,:,:,:,0], [8,10,15], 'encode_train.png', (0, 5))
	# mat2visual(encode_output_test[2,:,:,:,0], [8,10,15], 'encode_test.png', (0, 5))
	# mat2visual(decode_output_train[2,:,:,:,0], [40,45,60], 'decode_train.png',(-0.5, 3))
	# mat2visual(decode_output_test[2,:,:,:,0], [40,45,60], 'decode_test.png', (-0.5, 3))

def convAutoEncoderTest():
	fileInputs = sys.argv[1]
	inputSize = (91,109,91,1)
	cae = ConvolutionalAutoEncoder(32,(None,) + inputSize, (None,) + inputSize)
	convAeModel = cae.create_model()

	modelPath = 'conv_models/conv_autoencoder.hdf5'
	hf = h5py.File(fileInputs,'r')
	autInput = hf.get('swap_brain_autism')
	autInput = np.array(autInput)
	# autInput = HDF5Matrix(fileInputs, 'swap_brain_autism', start=0, end=2300)
	# autismOutput = cae.encode(autInput, modelPath)
	autismLabels = np.ones((2300,1))
	# print("Shape of the Autism output: {}".format(autismOutput.shape))
	print("Shape of the Autism labels: {}".format(autismLabels.shape))

	controlInput = hf.get('swap_brain_control')
	controlInput = np.array(controlInput)
	# controlInput = HDF5Matrix(fileInputs, 'swap_brain_control', start=0, end=2515)
	# controlOutput = cae.encode(controlInput, modelPath)
	controlLabels = np.zeros((2515,1))
	# print("Shape of the Control output: {}".format(controlOutput.shape))
	print("Shape of the Control labels: {}".format(controlLabels.shape))

	totData = np.vstack((autInput,controlInput))
	totLabels = np.vstack((autismLabels,controlLabels))

	numElements = totLabels.shape[0]
	indices = range(0,numElements)
	np.random.shuffle(indices)

	totData = totData[indices,:,:,:,:]
	totLabels = totLabels[indices,:]
	normalizedFmri = np.zeros((numElements,31, 37, 31, 1))

	for i in range(0,numElements):
		mu = np.mean(totData[i,:,:,:,0])
		var = np.std(totData[i,:,:,:,0])
		normalizedFmri[i,:,:,:,0] = (totData[i,:,:,:,0] - mu)/var

	print("Shape of totLabels: {}".format(totLabels.shape))
	print("Shape of data: {}".format(totData.shape))

	datHd = h5py.File('swap_brain_data_reduced.hdf5' , 'w')
	labHd = h5py.File('swap_brain_label_reduced.hdf5' , 'w')
	nomDatHd = h5py.File('swap_brain_data_reduced_normalized.hdf5' , 'w')

	datHd.create_dataset('swap_brain_fmri', data=totData, compression="gzip",  compression_opts=7)
	nomDatHd.create_dataset('swap_brain_fmri_normalized', data=normalizedFmri, compression="gzip",  compression_opts=7)
	labHd.create_dataset('swap_brain_label', data=totLabels, compression="gzip",  compression_opts=7)

def cnn_trainprocessing(filePath, where_start, where_end,dataset):
	hf = h5py.File(filePath,'r')
	matrix = HDF5Matrix(filePath, dataset, start=where_start, end=where_end)
	return matrix

def cnn_train():
	label=sys.argv[1]
	features=sys.argv[2]
	total = 4815
	numTrain = 4300

	x_train = cnn_trainprocessing(features,0, 4300,'swap_brain_fmri')

	y_train = cnn_trainprocessing(label,0, 4300,'swap_brain_label')

	x_val = cnn_trainprocessing(features,4300,4815,'swap_brain_fmri')

	y_val = cnn_trainprocessing(label,4300,4815, 'swap_brain_label')

	# # h = h5py.File(sys.argv[1],'r')
	# # autInput = h.get('swap_brain_label')
	# # autInput = np.array(autInput)
	# # print(autInput)
	# # print('reading')
	# # h = h5py.File(sys.argv[2],'r')
	# # print('getting')
	# # autInput = h.get('swap_brain_fmri')
	# # autInput = np.array(autInput)
	# print('obtaining')
	# # check=autInput[5]
	# a=x_train[250]-x_train[0]
	# print('printing')
	# print(a)



	print("Creating Convolutional Neural Network Object")
	batch_size = 32
	output_size = 1
	cnn = ConvolutionNeuralNetwork(batch_size, output_size)

	print("Defining the Convolutional AutoEncoder model")
	cnn_Model = cnn.create_model()

	print("Training the Convolutional Neural Network")
	numEpochs = 150

	train_log='convNeuralNetwork.log'


	cnn.train(x_train, y_train,x_val, y_val, numEpochs, train_log)


	modelPath = 'cnn_checkpoint.hdf5'
	train_predictions = cnn.test(x_train, modelPath)
	numIncorrect = np.sum(np.absolute(train_predictions - y_train))
	print("The training accuracy is {}".format(float(numTrain - numIncorrect)/float(numTrain)))

	#numTest = total - numTrain
	numTest=515
	test_predictions = cnn.test(x_val,modelPath)
	numIncorrect = np.sum(np.absolute(test_predictions - y_val))
	print("The testing accuracy is {}".format(float(numTest - numIncorrect)/float(numTest)))




if __name__ == "__main__":
	# convAutoEncoderTrain()
	# convAutoEncoderTest()
	cnn_train()

