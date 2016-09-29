# This file creates image data for the target dataset and target classes 
# by resizing to 256,256 and stores numpy array for train test and validation
# Uses antialiasing
import os
from os.path import join
import numpy as np
import PIL
from PIL import Image
# import gzip, cPickle
# import pickle
from sklearn.cross_validation import StratifiedShuffleSplit
#Root dir
rootDir = '/path-to-root-dir/' # where multiple type of datasets are stored
dataDir = '/path-to-data-dir/' # path where numpy would be stored
# Can work for all types of datasets. Just change in "if" condition.
# Python 3.0 does not have .next instead use in-built function
# datasetNames = os.walk(rootDir).next()[1]
datasetNames = next(os.walk(rootDir))[1]
# What classes to include in our images
targetClasses = ['class1','class2']
targetDataset = "" # Should be directory name
# Keras also work on tensorFlow as backend
backend = 'theano'
target_size = (256,256)
for dataset in datasetNames:
	if dataset==targetDataset:
		masterData = []
		labels = []
		imageList = []
		imageOriginalSize = []
		classDirPath = rootDir+dataset+'/'
		# classDirNames = os.walk(classDirPath).next()[1]
		classDirNames = next(os.walk(classDirPath))[1]
		for classDir in classDirNames:
			if classDir in targetClasses:
				imageDirPath = classDirPath+classDir+'/'
				imageNames = next(os.walk(imageDirPath))[2]
				# imageNames = os.walk(imageDirPath).next()[2]
				print(classDir)
				for imageName in imageNames:
					imageFile = imageDirPath+imageName
					try:
						img = Image.open(imageFile)
						imageOriginalSize.append(img.size)
						img=img.resize((target_size[1], target_size[0]), PIL.Image.ANTIALIAS)
						if backend == 'theano':
							imgSample = np.asarray(img, dtype='float32').transpose(2, 0, 1)
						else:
							imgSample = np.asarray(img, dtype='float32').transpose(2, 0, 1)
						masterData.append(imgSample)
						labels.append(classDir)
						imageList.append(imageName)
					except IOError:
						pass
		masterData = np.asarray(masterData)
		labels = np.asarray(labels)
		imageOriginalSize = np.asarray(imageOriginalSize)
		imageList = np.asarray(imageList)
		## Test here is a combination of validation and test
		indices = StratifiedShuffleSplit(labels, 1, test_size=1000, train_size=1500, random_state=0)
		for train_index, test_index in indices:
			masterDataTrain,masterDataTest = masterData[train_index],masterData[test_index]
			labelsTrain,labelsTest = labels[train_index],labels[test_index]
			imageOriginalSizeTrain, imageOriginalSizeTest = imageOriginalSize[train_index],imageOriginalSize[test_index]
			imageListTrain, imageListTest = imageList[train_index],imageList[test_index]
		np.save(dataDir+'train.npy',masterDataTrain)
		np.save(dataDir+'labelsTrain.npy',labelsTrain)
		np.save(dataDir+'imageOriginalSizeTrain.npy',imageOriginalSizeTrain)
		np.save(dataDir+'imageListTrain.npy',imageListTrain)
		### Split masterTest into test and validation
		indicesTest = StratifiedShuffleSplit(labelsTest, 1, test_size=0.5, random_state=0)
		for val_index, test_index in indicesTest:
			valid,testing = masterDataTest[val_index],masterDataTest[test_index]
			labelsValid,labelsTesting = labelsTest[val_index],labelsTest[test_index]
			imageOriginalSizeValid, imageOriginalSizeTesting = imageOriginalSizeTest[val_index],imageOriginalSizeTest[test_index]
			imageListValid, imageListTesting = imageListTest[val_index],imageListTest[test_index]
		np.save(dataDir+'validation.npy',valid)
		np.save(dataDir+'labelsValid.npy',labelsValid)
		np.save(dataDir+'imageOriginalSizeValid.npy',imageOriginalSizeValid)
		np.save(dataDir+'imageListValid.npy',imageListValid)
		##Save test dataset
		np.save(dataDir+'testing.npy',testing)
		np.save(dataDir+'labelsTesting.npy',labelsTesting)
		np.save(dataDir+'imageOriginalSizeTesting.npy',imageOriginalSizeTesting)
		np.save(dataDir+'imageListTesting.npy',imageListTesting)
