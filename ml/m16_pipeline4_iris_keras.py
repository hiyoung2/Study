# iris를 케라스 파이프라인 구성
# 당연히 RandomizedSearchCV 구성
# keras 98 참조

# 구성
# RandomizedSearchCV(pipe(전처리, 케라스모델), 파라미터, cv) #

import numpy as np

from sklearn.datasets import load_iris
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline, make_pipeline

iris = load_iris()
x = iris['data']
y = iris['target']


y = np_utils.to_categorical(y)
print("one-hot")
print("y.shape :", y.shape) # (150, 3)


# train_test_slit
x_train , x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 11, shuffle = True
)

print("=========================================")
print("x_train.shape :", x_train.shape)  # (120, 4)
print("x_test.shape :", x_test.shape)    # (30, 4)
print("y_train.shape :", y_train.shape)  # (120, 3)
print("y_test.shape :", y_test.shape)    # (30, 3)


def build_model(optimizer = 'adam', drop = 0.1) :
    
    inputs = Input(shape = (4, ), name = 'inputs')
    x = Dense(50, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(0.1)(x)
    x = Dense(70, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(0.1)(x)
    x = Dense(90, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(0.1)(x)
    x = Dense(110, activation = 'relu', name = 'hidden4')(x)
    x = Dropout(0.1)(x)
    x = Dense(10, activation = 'relu', name = 'hidden5')(x)
    outputs = Dense(3, activation = 'softmax', name = 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')

    return model

def create_hyperparameters() :
    batches = [125, 256, 512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5] # dropout에서 np.linspace가 안 먹힘, 그냥 리스트로 넣어줌
    # dropout = np.linspace(0.1, 0.5, 5).tolist() # 이렇게 하는 것도 가능! tolist() : list로 변환 해 준다
    epochs = [50, 100, 150, 200]
    return {"models__batch_size" : batches, "models__optimizer" : optimizers, 
             "models__drop" : dropout, "models__epochs" : epochs}

# 딕셔너리형으로 들어가는데
# pipeline에서 모델이름을 "model"로 명시해놨기 때문에
# 파라미터에 "model__"를 써 줘야 한다
# make_pipeline에서는 아마, "model__" 이라고 하면 돌아갈 듯?
# 선생님도 케라스에서 make_pipelinie 안 써봐서 모르심, 시도할 예정

model = KerasClassifier(build_fn = build_model, verbose = 1)

pipe = Pipeline([("scaler", MinMaxScaler()), ('models', model)])

# pipe = Pipeline([("scaler", MinMaxScaler()), ('model',model())])
# 이렇게 하니까 오류 발생
# 이미 model = Keras~ 위에 model은 무엇이다, 라고 정의를 해 놓았기 때문에
# ()를 쓸 필요가 없다
# m13 파일을 참고하면 
# pipe = make_pipeline(MinMaxScaler(), SVC())
# 이렇게 여기에선 모델명 다음에 ()를 썼는데 
# 이 때는 SVC에 대한 어떠한 것도 지정을 하지 않고
# from sklearn.svm import SVC
# import만 하고 있는 자체를 쓰는 셈이 된다
# 이 경우, 이 모델의 파라미터들이 그냥 디폴트값들로만 들어가 실행되는데
# 이 때는 ()를 써줘야 한다

# pipe = make_pipeline(MinMaxScaler(), model)

hyperparameters = create_hyperparameters()

kfold = KFold(n_splits = 5, shuffle = True)
search = RandomizedSearchCV(pipe, hyperparameters, cv = kfold)

from sklearn.metrics import accuracy_score


search.fit(x_train, y_train)

acc = search.score(x_test, y_test)

print("=========================================")
print("최적의 매개변수 :", search.best_estimator_)
print("=========================================")
print("최적의 매개변수 :", search.best_params_)
print("=========================================")
print("acc :", acc)

# y_pred = search.predict(x_test)

