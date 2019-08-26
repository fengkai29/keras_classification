from __future__ import print_function
from PIL import Image
import numpy as np

import tensorflow as tf
import grpc
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_shape_pb2


image_path = '/Users/fengkai/Desktop/pic/group1_M00_C9_3A_rBIeJVxiJkCAS50qAACQd51_-Y4270.jpg'

host = '192.168.23.90'
port = '8500'

image_size = (224,224,3)

# channel = implementations.insecure_channel(host=host,port=port)
channel = grpc.insecure_channel(host+":"+port)
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()

request.model_spec.name = 'nudeDetect'

request.model_spec.signature_name = 'keras_classification'

request.model_spec.version.value = 1

# test_datagen = ImageDataGenerator(
#     rescale=1/255.0,
# )

def deal_image(image_path,image_size):
        img = Image.open(image_path)
        img = img.resize((image_size[0], image_size[1]), Image.ANTIALIAS)
        img = np.array(img)
        img = np.multiply(img, 1.0 / 255.0)
        img = np.reshape(img, [1, img.shape[0], img.shape[1], image_size[2]])
        img = img.astype(np.float32)
        return img
image = deal_image(image_path,image_size)


# dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
#
# tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
# tensor_proto = tensor_pb2.TensorProto()
starttime = time.time()
request.inputs['image'].CopyFrom(tf.contrib.util.make_tensor_proto(image, shape=list(image.shape)))

result = stub.Predict(request, 60.0)
endtime = time.time()
result = tf.make_ndarray(result.outputs['classification'])
result = result[0].tolist()
print("-------------")
print(result.index(max(result)))
print(endtime-starttime)
print("-------------")