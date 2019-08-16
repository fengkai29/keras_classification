import tensorflow as tf
import os
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


# save_dir = './output/'
rawdata_root = '/Users/fengkai/PycharmProjects/keras_classification/dataset/data/train'

all_pd = pd.read_csv("/Users/fengkai/PycharmProjects/keras_classification/dataset/data/train.txt", sep=" ", header=None, names=['ImageName', 'label'])
train_pd, val_pd = train_test_split(all_pd, test_size=0.10, random_state=43, stratify=all_pd['label'])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

fc_size = 1000
nb_classes = 100

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
classes = 100

train_datagen = ImageDataGenerator(
     rotation_range=20,
     zoom_range=0.15,
     width_shift_range=0.2,
     height_shift_range=0.2,
     rescale=1/255.0,
     shear_range=0.15,
     horizontal_flip=True,
     fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(
    rescale=1/255.0,
)


if __name__ == '__main__':
    model_name = 'resnet50'
    train_gen = data_generator(rawdata_root, train_pd,IMAGE_SIZE, train_datagen,classes)
    val_gen = data_generator(rawdata_root, val_pd, IMAGE_SIZE, validation_datagen,classes)
    optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    exp_lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    base_model = ResNet50(weights = None, include_top = False,  input_shape=(224, 224, 3))
    #model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=100)
    #model.summary()
    # load the checkpoint
    save_path = os.path.join('trained_model', model_name)
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)
    model_names = (os.path.join(save_path, model_name + '.{epoch:02d}-{val_acc:.4f}.hdf5'))
    resume = None
    if resume:
        print("load weight resume")
        base_model.load_weights(resume)
    model = add_new_last_layer(base_model, fc_size, nb_classes)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy',fmeasure,recall,precision])
    checkpoint = ModelCheckpoint(model_names,monitor='val_acc',verbose=1,save_best_only=True,period = 5)
    # print(np.asarray(train_gen.images).shape)
    # print(np.asarray(train_gen.labels).shape)
    model.fit_generator(generator=train_datagen.flow(train_gen.images,train_gen.labels,batch_size=4,shuffle=True), steps_per_epoch=1000,epochs=50, verbose= 1,  validation_data = validation_datagen.flow(val_gen.images,val_gen.labels,batch_size=4,shuffle=True),validation_steps = 10,callbacks=[checkpoint,exp_lr_scheduler],initial_epoch=0)