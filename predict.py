import tensorflow as tf
import os
import math
import pandas as pd
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from model.resnet50 import ResNet50
#import model.model as md
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications import resnet50
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from dataset.measure import fmeasure,recall,precision
from dataset.generrator import data_generator
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Dropout
from keras import layers
from keras.applications.vgg16 import VGG16
from keras.models import Model,load_model

resume = None
rawdata_root = '/home/kai.feng/pytorch_classification/dataset/train_improve/data_all'

all_pd = pd.read_csv("/home/kai.feng/pytorch_classification/dataset/train_improve/train.txt", sep=" ", header=None, names=['ImageName', 'label'])
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
fc_size = 1000
nb_classes = 2

def add_new_last_layer(base_model, FC_SIZE, nb_classes):
    """
    添加最后的层
    输入
    base_model和分类数量
    输出
    新的keras的model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


width,height=224,224
IMAGE_SIZE=(width,height,3)
classes = 2

test_datagen = ImageDataGenerator(
    rescale=1/255.0,
)

if __name__ == '__main__':
    test_gen = data_generator(rawdata_root, all_pd, IMAGE_SIZE, test_datagen, classes)
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = add_new_last_layer(base_model, fc_size, nb_classes)
    model.load_weights(resume)
    predicts = model.predict_generator(test_datagen.flow(test_gen.images,batch_size=4),
                                       steps=math.ceil(len(test_gen.labels) / 4),
                                       # callbacks=None,
                                       # max_queue_size=10,
                                       # workers=1,
                                       # use_multiprocessing=False,
                                       verbose=1)
    count = 0
    for i in range(len(predicts)):
        if predicts[i]==test_gen.labels[i]:
            count = count+1

    print("accuracy: %.2f" %(count/len(predicts)))