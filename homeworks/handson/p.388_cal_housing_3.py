# 보조 출력 추가
# 적절한 층에 연결하고 모델의 출력 리스트에 추가하면 된다


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full
)

print("X_train.shape :", X_train.shape) # (11610, 8)
print("X_train.shape[1:]", X_train.shape[1:]) # (8,)
print("y_train.shape :", y_train.shape) # (11610,)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

import keras

input_A = keras.layers.Input(shape = [5], name = "wide_input")
input_B = keras.layers.Input(shape = [6], name = "deep_input")
hidden1 = keras.layers.Dense(30, activation = 'relu')(input_B)
hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)
concat = keras.layers.concatenate([input_A, hidden2]) 
output = keras.layers.Dense(1, name = 'outpupt')(concat) # 출력층까지는 이전과 동일 
aux_output = keras.layers.Dense(1, name = 'aux_output')(hidden2)
model = keras.Model(inputs = [input_A, input_B], outputs = [output, aux_output])

# 각 출력은 자신만의 손실 함수가 필요, 따라서 모델을 컴파일할 때 손실의 리스트를 전달해야 한다
# 하나의 손실을 전달하면 케라스는 모든 출력의 손실 함수가 동일하다고 가정한다
# 기본적으로 케라스는 나열된 손실을 모두 더하여 최종 손실을 구해 훈련에 사용한다
# 보조 출력보다 주 출력에 더 관심이 많다면(보조 출력은 규제로만 사용되므로), 주 출력의 손실에 더 많은 가중치를 부여해야한다
# 다행히 모델을 컴파일 할 때 손실 가중치를 지정할 수 있다

model.compile(loss = ["mse", "mse"], loss_weights = [0.9, 0.1], optimizer = "sgd")

# 이제 모델을 훈련할 때 각 출력에 대한 레이블을 제공해야 한다
# 여기에서는 주 출력과 보조 출력이 같은 것을 예측해야하므로 동일한 레이블을 사용한다
# 따라서 y_train 대신에 (y_train, y_train)을 전달한다(y_valid, y_test 도 동일)

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs = 20, validation_data = ([X_valid_A, X_valid_B], [y_valid, y_valid])
)

# 모델을 평가하려면 케라스는 개별 손실과 함께 총 손실을 반환한다
total_loss, main_loss, aux_loss = model.evaluate(
    [X_test_A, X_test_B], [y_test, y_test]
)
print("total loss :", total_loss)
print("main loss :", main_loss)
print("aux loss :", aux_loss)

'''
total loss : 0.3787951306317204
main loss : 0.36283519864082336
aux loss : 0.51121586561203
'''

# predict() 메서든느 각 출력에 대한 예측을 반환
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

print("Y_pred.main:", y_pred_main)
print("Y_pred.aux:", y_pred_aux)

'''
Y_pred.main: [[3.0434155]
 [2.041671 ]
 [1.6575624]]
Y_pred.aux: [[2.9177752]
 [2.9309633]
 [1.6690227]]
'''