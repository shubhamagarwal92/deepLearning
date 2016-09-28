## vgg16 model taken from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
# Be careful about validation in keras. Developer of keras has used validation to give 
# accuracy results. Not used in training.
# Vgg model modified according to the use case. Drop outs and number of layers changed.
# Can even change activations

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
import pandas as pd
from keras.callbacks import ModelCheckpoint
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

##Parmeter Initialize
img_width, img_height = 256, 256
rootDir = '/path-to-dir/'
dataDir = rootDir+'data/'
nb_train_samples = 512
##samples per epoch should be a exact divisor of batch size. Otherwise warning.
nb_validation_samples = 256
nb_epoch = 10
nb_test_samples = 256
### Model - VGG 16
first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
model = Sequential()
model.add(first_layer)
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Convolutional - 2
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Convolutional - 3
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Convolutional - 4
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Convolutional Layer 5
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Flatten and add dropout before sigmoid layer.
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# For Multi class
# model.add(Dense(num_classes, activation='softmax'))
# Optimiser
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# Compile the model with parameters as loss,optimiser,metrics
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# this is the augmentation configuration we will use for training. For now we are using mean centering
train_datagen = ImageDataGenerator(
        rescale=1./255,
samplewise_center=True)
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
#Load Train
x_train = np.load(dataDir+'train.npy')
y_train = np.load(dataDir+'labelsTrain.npy')
#Load Validation
x_val = np.load(dataDir+'validation.npy')
y_val = np.load(dataDir+'labelsValid.npy')
#Load Test
x_test = np.load(dataDir+'testing.npy')
y_test = np.load(dataDir+'labelsTesting.npy')
#Encode once using sklearn encoder for class labels
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)
# np.bincount(y_train)
np.unique(y_train,return_counts=True)
# For One hot encoding use:
# dummy_y = np_utils.to_categorical(encoded_Y)
##
## Create data flow for Keras
train_generator = train_datagen.flow(
    x_train,y_train,
    batch_size=32)
valid_generator = val_datagen.flow(
    x_val,y_val,
    batch_size=32)
# test_generator = test_datagen.flow(
#     x_test,y_test,
#     batch_size=32)
history = LossHistory()

model.fit_generator(
        train_generator,
        samples_per_epoch	=nb_train_samples,
        nb_epoch			=nb_epoch,
        validation_data		=valid_generator,
        nb_val_samples 		=nb_validation_samples,
        verbose             =1,
        callbacks           =[history]))
# #Print accuracy on test
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (score[1]*100))
# Validation Prediction
y_val_pred = model.predict(x_val)
# Identifying optimal threshold
fpr,tpr,thresh = metrics.roc_curve(y_val,np.array(y_val_pred),1)
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr-(1-fpr), index = i), 'thresholds' : pd.Series(thresh, index = i)})
th= roc.ix[(roc.tf-0).abs().argsort()[:1]]
threshold =  np.float(th.thresholds)
func = np.vectorize(lambda x: 1 if x > threshold else 0)
# Predictions on test data set
y_pred = model.predict(x_test)
test_pred = func(y_pred)
print(classification_report(y_test, test_pred))
print(metrics.confusion_matrix(y_test, test_pred))
print(metrics.accuracy_score(y_test, test_pred))
#### Saving model
# serialize model to JSON
model_json = model.to_json()
modelDir = rootDir + 'models/' 
modelFile = modelDir + "model.json"
with open(modelFile, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelDir+"vgg16.h5")
print("Saved model to disk")
