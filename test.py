# import numpy as np
#
# np.random.seed(1337)
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import RMSprop
#
# # Download the mnist to the path ~/.keras/datasets/ if it is the first time to be called
# # XShape(6000 28X28),y shape(10,000,)
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(type(X_train))
# print(X_test[0])
# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# print(y_test[0])
# # Data pre-processing
# X_train = X_train.reshape(X_train.shape[0], -1) / 255  # Normalize
# X_test = X_test.reshape(X_test.shape[0], -1) / 255  # Normalize
# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)
#
# # Other way to build your neural net
# model = Sequential([
#     Dense(32, input_dim=784),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax')
# ])
#
# # Other way to define optimizer
# rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# # model.compile(optimizer='rmsprop')
# model.compile(
#     optimizer='rmsprop',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# print('Training..............')
# # Other way to train the model
# model.fit(X_train, y_train, epochs=2, batch_size=32)
#
# print('\nTesting..............')
# loss, accuracy = model.evaluate(X_test, y_test)
# print('\nTest lost:', loss)
# print('Test accuracy', accuracy)

a = [1,2]
b = [1,2]
c = []
c.append(a)
c.append(b)
print(c.shape())