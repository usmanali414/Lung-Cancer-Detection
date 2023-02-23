# import tensorflow as tf
# # import tensorflow
# from tensorflow import keras
# from keras.preprocessing import image
# from keras.models import Model
# from keras.layers import Dense, Dropout, Flatten
# from keras import backend as K
# from keras.models import Model
# from keras.layers import Input,Dense,Activation,Dropout,GlobalAveragePooling2D
# from tensorflow.keras.applications import ResNet50
# import tensorflow.keras.applications
# from keras.applications.resnet50 import preprocess_input
# from keras.optimizers import Adam,SGD
# from metrics import *



# def get_resnet50():
#   resnet50_model = ResNet50(include_top = False,input_shape=(512,512,3), weights = 'imagenet')
#   x = resnet50_model.output
#   x = GlobalAveragePooling2D()(x)
#   x = Dropout(0.2)(x)
#   x = Dense(512)(x)
#   x = Activation('relu')(x)
#   x = Dense(1)(x)
#   x = Activation('sigmoid')(x)
#   for layer in resnet50_model.layers:
#     layer.trainable = False
#   resnet50_model = Model(inputs=resnet50_model.inputs, outputs=x)
# 	#adam = Adam(lr=0.001, decay=1e-6)
#   sgd = SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
#   resnet50_model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy',recall_m,precision_m,f1_m])
#   return resnet50_model

# def get_vgg19():

#   vgg19 = tensorflow.keras.applications.vgg19
#   conv_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(512,512,3))
#   for layer in conv_model.layers: 
#       layer.trainable = False
#   x = tensorflow.keras.layers.Flatten()(conv_model.output)
#   x = tensorflow.keras.layers.Dense(100, activation='relu')(x)
#   x = tensorflow.keras.layers.Dense(100, activation='relu')(x)
#   x = tensorflow.keras.layers.Dense(100, activation='relu')(x)
#   predictions = tensorflow.keras.layers.Dense(1, activation='sigmoid')(x)
#   full_model = tensorflow.keras.models.Model(inputs=conv_model.input, outputs=predictions)
#   #adam = Adam(lr=0.0001, decay=1e-6)
#   sgd = SGD(learning_rate=0.001,momentum=0.9,nesterov=True)
#   full_model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy',recall_m,precision_m,f1_m])
#   #full_model.summary()
  
#   return full_model
import tensorflow
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
#from keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam,SGD
from Model_scripts.metrics import *
import os


# 
def get_xception(checkstatus='train'):
    if checkstatus=='train':
      xception_model = tensorflow.keras.applications.Xception(include_top=False,
                                                      weights="imagenet",
                                                      input_tensor=None,
                                                      input_shape=(512,512,3),
                                                      pooling='avg')
    else:
      xception_model = tensorflow.keras.applications.Xception(include_top=False,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=(512,512,3),
                                                pooling='avg')    
                                                    
    x = xception_model.output
    x = Dense(512, activation='relu',kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    x = Dense(512, activation='relu',kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation = 'sigmoid',kernel_regularizer=tensorflow.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)) (x)
    for layer in xception_model.layers:
        layer.trainable = False
    model = Model(inputs=xception_model.input, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.Adamax(lr=0.00001),
                metrics=['accuracy',recall_m,precision_m,f1_m])
    return model

def get_vgg16():

  base_model = VGG16(weights='imagenet', include_top=False, input_shape = (256, 256, 3))
  x = base_model.output
  x = Flatten(input_shape=base_model.output_shape[1:])(x)
  x = Dense(2048, activation='relu',kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
  x = Dropout(0.4)(x)
  # and a logistic layer -- let's say we have 200 classes
  predictions = Dense(1, activation = 'sigmoid',kernel_regularizer=tensorflow.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)) (x)

  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)

  # set the first 25 layers (up to the last conv block)
  # to non-trainable (weights will not be updated)
  for layer in model.layers[:19]:
    layer.trainable = False
  adam = Adam(lr=0.0001, decay=1e-6)
  sgd = SGD(learning_rate=0.00001,momentum=0.9,nesterov=True)
  model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy',recall_m,precision_m,f1_m])
  return model

def get_inceptionv3():

    inception_model = InceptionV3(include_top=False,
                                                    weights="imagenet",
                                                    input_tensor=None,
                                                    input_shape=(512,512,3),
                                                    pooling=None)
    x = inception_model.output
    x = Flatten(input_shape=inception_model.output_shape[1:])(x)
    x = Dense(512, activation='relu',kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    x = Dense(512, activation='relu',kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation = 'sigmoid',kernel_regularizer=tensorflow.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)) (x)
    for layer in inception_model.layers:
        layer.trainable = False
    model = Model(inputs=inception_model.input, outputs=predictions)
    #adam = Adam(lr=0.0001, decay=1e-6)
    #sgd = SGD(learning_rate=0.00001,momentum=0.9,nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.Adamax(lr=0.0001),
                metrics=['accuracy',recall_m,precision_m,f1_m])
    return model

# def get_inceptionv3():
#   inception_model = InceptionV3(include_top=False,
#                                                  weights="imagenet",
#                                                  input_tensor=None,
#                                                  input_shape=(299,299,3),
#                                                  pooling=None)
#   x = inception_model.output
#   x = Flatten(input_shape=inception_model.output_shape[1:])(x)
#   x = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x)
#   #x = Dense(512, activation='relu')(x)#,kernel_regularizer=keras.regularizers.l2(0.001))(x)
#   x = Dropout(0.4)(x)
#   # and a logistic layer -- let's say we have 200 classes
#   predictions = Dense(1, activation = 'sigmoid',kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)) (x)
#   #predictions = Dense(1, activation = 'sigmoid')(x)
#   for layer in inception_model.layers:
#     layer.trainable = False
#   # this is the model we will train
#   model = Model(inputs=inception_model.input, outputs=predictions)

#   # set the first 25 layers (up to the last conv block)
#   # to non-trainable (weights will not be updated)

#   adam = Adam(lr=0.00001, decay=1e-6)
#   sgd = SGD(learning_rate=0.00001,momentum=0.9,nesterov=True)
#   model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy',recall_m,precision_m,f1_m])
#   return model






  # vgg16 = keras.applications.vgg16
  # base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = (256, 256, 3))
  # x = keras.layers.Flatten()(base_model.output)
  # x = keras.layers.Dense(1024, activation='relu')(x)
  # x = keras.layers.Dropout(0.6)(x)
  
  # predictions = keras.layers.Dense(1, activation = 'sigmoid')(x)

  # # this is the model we will train
  # model = Model(inputs=base_model.input, outputs=predictions)
  # # set the first 25 layers (up to the last conv block)
  # # to non-trainable (weights will not be updated)
  # for layer in model.layers[:19]:
  #     layer.trainable = False
  # sgd = SGD(learning_rate=0.0001,momentum=0.9,nesterov=True)
  # model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy',recall_m,precision_m,f1_m])
  # return model
  # vgg16 = tensorflow.keras.applications.vgg16
  # base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = (256, 256, 3))
  # x = tensorflow.keras.layers.Flatten()(base_model.output)
  # x = tensorflow.keras.layers.Dense(1024, activation='relu')(x)
  # x = tensorflow.keras.layers.Dropout(0.6)(x)
  
  # predictions = tensorflow.keras.layers.Dense(1, activation = 'sigmoid')(x)

  # # this is the model we will train
  # model = Model(inputs=base_model.input, outputs=predictions)

  # # set the first 25 layers (up to the last conv block)
  # # to non-trainable (weights will not be updated)
  # for layer in model.layers[:19]:
  #     layer.trainable = False

  # sgd = SGD(learning_rate=0.0001,momentum=0.9,nesterov=True)
  # model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy',recall_m,precision_m,f1_m])
  # return model