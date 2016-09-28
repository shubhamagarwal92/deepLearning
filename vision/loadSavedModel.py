from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import classification_report
import h5py
from keras import models
from keras.models import model_from_json
import pandas as pd
##Parmeter Initialize
img_width, img_height = 256, 256
rootDir = '/path-to-dir/'
dataDir = rootDir+'data/'
nb_train_samples = 512
##samples per epoch should be a exact divisor of batch size
nb_validation_samples = 256
nb_epoch = 10
nb_test_samples = 256
#Directories
modelDir = rootDir + 'models/' 
modelFile = modelDir + "model.json"
weightFile = modelDir + "vgg16.h5"
# Load JSON object and weights
json_file = open(modelFile, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weightFile)
print("Loaded model from disk")
model = loaded_model

# Further do processing 

