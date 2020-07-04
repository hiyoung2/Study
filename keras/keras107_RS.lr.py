# 100번 파일 copy, lr과 optimizer를 넣고 튜닝
# lr 적용 : 숫자라 np.linspace를 쓰고 리스트로 바꿔줘야 하므로 .tolist()가 필요하다
# optimizer 적용 : import 받아서 randomsearch의 parameter에 list 형태로 넣어줘야 한다
# LSTM -> Dense로 바꿀 것

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils 
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

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

def build_model(drop=0.5, optimizer = 'Adam', learning_rate = 0.01) :
    inputs = Input(shape = (28*28, ), name = 'input')
    x = Dense(128, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(64, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = optimizer(learning_rate = learning_rate), metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() :
    batches = [32, 128, 256]
    optimizers = [Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam]
    # learning_rate = np.linspace(0.1, 0.9, 9).tolist()
    learning_rate = [0.01, 0.03, 0.05, 0.07, 0.09]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    return {"batch_size" : batches, "optimizer" : optimizers, "learning_rate" : learning_rate, 
            "drop" : dropout} # dictionary 형태

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
최적의 파라미터 : {'optimizer': <class 'keras.optimizers.Adagrad'>, 'learning_rate': 0.03, 'drop': 0.30000000000000004, 'batch_size': 128}
ACC : 0.9458000063896179
'''