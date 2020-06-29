# 진동데이터 코드공유

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def kaeri_metric(y_true, y_pred) :
    '''
    y_true: dataframe with true values of X, Y, M, V
    y_pred: dataframe with pred values of X, Y, M, V

    return: KAERI metric
    '''
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)

# E1과 E2는 아래에 정의됨
def E1(y_true, y_pred) :
    '''
    y_true: dataframe with true values of X, Y, M, V
    y_pred: dataframe with pred values of X, Y, M, V
    return: distance error normalized with 2e+04
    '''
    _t, _p = np.array(y_true[:, :2], np.array(y_pred[:, :2]))

    return np.mean(np.sum(np.square(_t - _p), axis = 1 / 2e+04))

def E2(y_true, y_pred) :
    '''
    y_ture: dataframe with true values of X, Y, M, V
    y_pred: dataframe with pred values of X, Y, M, V

    return: sum of mass and velocity's mean squared percentage error
    '''

    _t, _p = np.array(y_true)[:, 2:], np.array(y_pred[:, 2:])

    return np.maen(np.sum(np.square(_t - _p) / (_t + 1e-06)), axis = 1)

import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Lambda, AveragePooling2D
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


# import jovian
# import kaeri_metric
# from kaeri_metric import E1, E2, kaeri_metric, E2M, E2V

# x_data = []
# y_data = []

x_data = np.loadtxt('./data/dacon/comp3/train_features.csv', skiprows = 1, delimiter = ',')
x_data = x_data[:, 1:]
print("x_data.shape :", x_data.shape) # (1050000, 5)


y_data = np.loadtxt('./data/dacon/comp3/train_target.csv', skiprows = 1, delimiter = ',')
y_data = y_data[:, 1:]
print("y_data.shape :", y_data.shape) # (2800, 4)

x_data = x_data.reshape((2800, 375, 5, 1))
print("x_data.shape :", x_data.shape) # (2800, 375, 5, 1)

x_data_test = np.loadtxt("./data/dacon/comp3/test_features.csv", skiprows = 1, delimiter = ',')
x_data_test = x_data_test[:, 1:]
x_data_test = x_data_test.reshape((700, 375, 5, 1))



data_id = 2

plt.figure(figsize = (8, 6))

plt.plot(x_data[data_id, :, 0, 0], label = "Sensor #1")
plt.plot(x_data[data_id, :, 1, 0], label = "Sensor #2")
plt.plot(x_data[data_id, :, 2, 0], label = "Sensor #3")
plt.plot(x_data[data_id, :, 3, 0], label = "Sensor #4")

plt.xlabel("Time", labelpad = 10, size = 20)
plt.ylabel("Acceleration", labelpad = 10, size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlim(0, 400)
plt.legend(loc = 1)


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2
    )

plt.show()

weight1 = np.array([1, 1, 0, 0])
weight2 = np.array([0, 0, 1, 1])

def my_loss(y_true, y_pred) :
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true), (y_true+0.000001)])
    return K.mean(K.square(divResult))

def my_loss_E1(y_true, y_pred) :
    return K.mean(K.square(y_true - y_pred)*weight1)/2e+04

def my_loss_E2(y_true, y_pred) :
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true), (y_true+0.000001)])
    return K.mean(K.squre(divResult)*weight2)


# tr_target = 2

def set_model(train_target): # 0 : x, y, 1 : m, 2 : v

    activation = 'elu'
    padding = 'valid'
    model = Sequential()
    nf = 16
    fs = (3, 1)

    model.add(Conv2D(nf, fs, padding = padding, activation = activation, input_shape = (375, 5, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 1)))

    model.add(Conv2D(nf*2, fs, padding = padding, activation = activation, input_shape = (375, 5, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 1)))

    model.add(Conv2D(nf*4, fs, padding = padding, activation = activation, input_shape = (375, 5, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 1)))
    
    model.add(Conv2D(nf*8, fs, padding = padding, activation = activation, input_shape = (375, 5, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 1)))

    model.add(Conv2D(nf*16, fs, padding = padding, activation = activation, input_shape = (375, 5, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 1)))

    model.add(Conv2D(nf*32, fs, padding = padding, activation = activation, input_shape = (375, 5, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 1)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'elu'))
    model.add(Dense(64, activation = 'elu'))
    model.add(Dense(32, activation = 'elu'))
    model.add(Dense(16, activation = 'elu'))
    model.add(Dense(4))

    optimizer = keras.optimizers.Adam()


    global weight2
    if train_target == 1 : # only for M
        weight2 = np.array([0, 0, 1, 0])
    else : # only for V
        weight2 = np.array([0, 0, 0, 1])


    if train_target == 0 : 
        model.compile(loss = my_loss_E1, optimizer = optimizer)

    else :
        model.compile(loss = my_loss_E2, optimizer = optimizer)

    model.summary()

    return model

# 학습 및 모델 리턴

def train(model,X,Y):
    MODEL_SAVE_FOLDER_PATH = './model/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    best_save = ModelCheckpoint('best_m.hdf5', save_best_only=True, monitor='val_loss', mode='min')


    history = model.fit(X, Y,
                  epochs=100,
                  batch_size=256,
                  shuffle=True,
                  validation_split=0.2,
                  verbose = 2,
                  callbacks=[best_save])

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    plt.show()    
    
    return model


# error plot

def plot_error(type_id, pred, true) :
    print(pred.shape)

    if type_id == 0 :
        _name = 'x_pos'
    elif type_id == 1 :
        _name = 'y_pos'
    elif type_id == 2 :
        _name = 'mass'
    elif type_id == 3 :
        _name = 'velocity'
    elif type_id == 4 :
        _name = 'distance'
    else :
        _name = 'error'

    x_coord = np.arange(1, pred.shape[0]+1, 1)
    if type_id < 2 :
        Err_m = (pred[:, type_id] - true[:, type_id])
    elif type_id < 4 :
        Err_m = ((pred[:, type_id] - true[:, type_id])/true[:, type_id])*100
    else :
        Err_m = ((pred[:, 0] - true[:, 0])**2 + (pred[:, 1] - true[:, 1])**2)**0.5

        
    

    fig = plt.figure(figsize = (8, 6))

    # plt.rcParams["font.family"] = "This New Roman"
    plt.rcParams["font.size"] = 15
    plt.scatter(x_coord, Err_m, marker = 'o')
    plt.title("%s Prediction for Training Data" % _name, size = 20)
    plt.xlabel("Data ID", labelpad = 10, size = 20)
    plt.ylabel("Prediction Error of %s" % _name, labelad = 10, size = 20)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.ylim(-100., 100.)
    plt.xlim(0, pred.shape[0]+1)

    plt.show()

    print(np.std(Err_m))
    print(np.max(Err_m))
    print(np.min(Err_m))
    return Err_m

# plot_error(type_id, pred, true) :



# Best Model Load

def load_best_model(train_target) :

    if train_target == 0 :
        model = load_model('best_m.hdf5', custom_objects = {'my_loss_E1' : my_loss, })
    else :
        model = load_model('best_m.hdf5', custom_objects = {'my_loss_E2' : my_loss, })
    
    score = model.evaluate(x_data, y_data, verbose = 0)
    print('loss :', score)

    pred = model.predict(x_data)

    i = 0

    print('정답(original) :', y_data[i])
    print('예측값(original) :', pred[i])

    print(E1(pred, y_data))
    print(E2(pred, y_data))
    # print(E2M(pred, y_data))
    # print(E2V(pred, y_data))

    if train_target == 0 :
        plot_error(4, pred, y_data)
    elif train_target == 1 :
        plot_error(2, pred, y_data)
    elif train_target == 2:
        plot_error(3, pred, y_data)

    return model

submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')

for train_target in range(3) :
    model = set_model(train_target)
    train(model, x_train, y_train)
    best_model = load_best_model(train_target)

    pred_data_test = best_model.predict(x_data_test)

    if train_target == 0 : # x, y 학습
        submit.iloc[:, 1] = pred_data_test[:, 0]
        submit.iloc[:, 2] = pred_data_test[:, 1]

    elif train_target == 1 : # m 학습
        submit.iloc[:, 3] = pred_data_test[:, 2]
    
    elif train_target == 2 : # v 학습
        submit.iloc[:, 4] = pred_data_test[:, 3]


submit.to_csv('./dacon/comp3/submission/0629/submit_0629_h', index = False)