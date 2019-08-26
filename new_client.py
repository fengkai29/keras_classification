from __future__ import print_function
from PIL import Image
import numpy as np
import tensorflow as tf
import grpc
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2


image_path = '/Users/fengkai/Desktop/pic/group1_M00_C9_3A_rBIeJVxiJkCAS50qAACQd51_-Y4270.jpg'

host = '192.168.23.90'
port = '9500'
image_size = (224,224,3)

# channel = implementations.insecure_channel(host=host,port=port)
channel = grpc.insecure_channel(host+":"+port)
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()

request.model_spec.name = 'nudeDetect'

request.model_spec.signature_name = 'keras_classification'

request.model_spec.version.value = 1

def deal_image(image_path,image_size):
        img = Image.open(image_path)
        img = img.resize((image_size[0], image_size[1]), Image.ANTIALIAS)
        img = np.array(img)
        img = np.multiply(img, 1.0 / 255.0)
        img = np.reshape(img, [1, img.shape[0], img.shape[1], image_size[2]])
        img = img.astype(np.float32)
        return img
img = deal_image(image_path,image_size)
# print(list(img.reshape(-1)))
starttime = time.time()
dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in list(img.shape)]
tensor = tensor_pb2.TensorProto(
              dtype=types_pb2.DT_FLOAT,
              tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=dims),
              float_val=list(img.reshape(-1)))
request.inputs['image'].CopyFrom(tensor)


result = stub.Predict(request, 10.0)
endtime = time.time()
result = tf.make_ndarray(result.outputs['classification'])
result = result[0].tolist()
print("-------------")
print(result.index(max(result)))
print(endtime-starttime)
print("-------------")