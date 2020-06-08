# 98번 -> LSTM 모델로 

from keras.datasets import mnist
from keras.utils import np_utils 
from keras.models import Sequential, Model
from keras.layers import Input, Dropout
from keras.layers import Dense, LSTM
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape :", x_train.shape) 
print("x_test.shape : ", x_test.shape) 
print("y_train.shape :", y_train.shape) 
print("y_test.shape :", y_test.shape)  


# 1. 데이터
x_train = x_train.reshape(x_train.shape[0], 28, 28)/255
x_test = x_test.reshape(x_test.shape[0], 28, 28)/255

print("x_train.reshape :", x_train.shape)  # (60000, 28, 28)
print("x_test.reshape :", x_test.shape)    # (10000, 28, 28)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train.shape.oh :", y_train.shape) # (60000, 10)
print("y_test.shape.oh :", y_test.shape)   # (10000, 10)


# 2. 모델 구성

def build_model(drop=0.5, optimizer = 'adam') :
    inputs = Input(shape = (28, 28), name = 'inputs')
    x = LSTM(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() :
    batchs = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5) 
    return {"batch_size" : batchs, "optimizer" : optimizers, 
            "drop" : dropout} # dictionary 형태

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters() 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv = 3)

# 3. 훈련(실행)
search.fit(x_train, y_train)
acc = search.score(x_test, y_test)

print("최적의 파라미터 :", search.best_params_)
print("acc :", acc)