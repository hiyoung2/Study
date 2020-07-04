# 107번에 activation을 넣어서 완성

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils 
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
from keras.layers import LeakyReLU
from keras.activations import relu, elu, selu, sigmoid, softmax, tanh
leaky = LeakyReLU(0.2)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape :", x_train.shape) # (60000, 28, 28)
print("x_test.shape : ", x_test.shape)  # (10000, 28, 28)
print("y_train.shape :", y_train.shape) # (60000,)
print("y_test.shape :", y_test.shape)   # (10000,)


# 1-1 데이터 전처리
x_train = x_train.reshape(x_train.shape[0], 28*28)/255
x_test = x_test.reshape(x_test.shape[0], 28*28)/255

print("x_train.reshape :", x_train.shape)  # (60000, 28, 28)
print("x_test.reshape :", x_test.shape)    # (10000, 28, 28)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train.shape.oh :", y_train.shape) # (60000, 10)
print("y_test.shape.oh :", y_test.shape)   # (10000, 10)


# 2. 모델 구성

def build_model(drop=0.5, optimizer = 'Adam', learning_rate = 0.01, activation = 'relu') :
    inputs = Input(shape = (28*28, ), name = 'input')
    x = Dense(256, activation = activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation = activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation = activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = optimizer(learning_rate = learning_rate), metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() :
    batches = [128, 256, 512]
    optimizers = [Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam]
    # learning_rate = np.linspace(0.1, 0.9, 9).tolist()
    learning_rate = [0.01, 0.03, 0.05, 0.07, 0.09]
    # dropout = np.linspace(0.1, 0.5, 5).tolist()
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'tanh', 'leaky']
    #sigmoid, softmax, tanh, relu, leakyrelu, elu, selu
    return {"batch_size" : batches, "optimizer" : optimizers, "learning_rate" : learning_rate, 
            "drop" : dropout, "activation" : activation} # dictionary 형태

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters() 


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv = 3)

# 3. 훈련(실행)
search.fit(x_train, y_train)
acc = search.score(x_test, y_test)

print("최적의 파라미터 :", search.best_params_)
print("ACC :", acc)

'''
최적의 파라미터 : {'optimizer': <class 'keras.optimizers.Adagrad'>, 'learning_rate': 0.01, 'drop': 0.1, 'batch_size': 256, 'activation': <function elu at 0x0000027249610828>}
ACC : 0.9355000257492065
'''

'''
최적의 파라미터 : {'optimizer': <class 'keras.optimizers.Adagrad'>, 'learning_rate': 0.05, 'drop': 0.4, 'batch_size': 256, 'activation': 'selu'}
ACC : 0.8682000041007996
'''