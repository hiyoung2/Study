# 10.2.7 콜백 사용ㅇ하기

# fit() 메서드의 callbacks 매개변수를 사용하여 케라스가 훈련의 시작이나 끝에 호출할 객체 리스트를 지정할 수 있다
# 또는 에포크의 시작이나 끝, 각 배치 처리 전후에 호출할 수도 있다
# 예를 들어 ModelCheckpoint는 훈련하는 동안 일정한 간격으로 모델의 체크포인트를 저장한다
# 기본적으로 매 에포크의 끝에서 호출된다

# [...] # 모델을 만들고 컴파일하기(생략)
# checkpoint_cb = keeras.callbacks.ModelCheckpoint("my_keras_model.h5")
# history = model.fit(X_train, y_train, epochs = 10, callbacks = [checkpoint_cb])

# 훈련하는 동안 검증 세트를 사용하면 ModelCheckpoint를 만들 때 save_best_only = True로 지정할 수 있다
# 이렇게 하면 최상의 검증 세트 점수에서만 모델을 저장한다!
# 오랜 시간으로 훈련 세트에 과대적합될 걱정을 하지 않아도 된다
# 훈련이 끝난 후 마지막에 저장된 모델을 복원하면 된다
# 그 모델이 검증 세트에서 최상의 점수를 낸 모델이다

# 다음 코드는 조기 종료를 구현하는 방법이다

# checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
# history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_valid, y_valid), callbacks = [checkpoint_cb])

# model = keras.models.load_model("my_keras_model.h5") # 최상의 모델로 복원

# 조기 종료를 구현하는 또 다른 방법은 EarlyStopping 콜백을 사용하는 것이다
# 일정 에포크(patience 매개변수로 지정) 동안 검증 세트에 대한 점수가 향상되지 않으면 훈련을 멈춘다
# 선택적으로 최상의 모델을 복원할 수도 있다
# (컴퓨터가 문제를 일으킬 경우를 대비해서) 체크포인트 저장 콜백과 (시간과 컴퓨터 자원을 낭비하지 않기 위해) 진전이 없는 경우 훈련을 일찍 멈추는 콜백을 함께 사용할 수 있다

# early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
# history = mode.fit(X_train, y_train, epochs = 100, validation_data = (X_valid, y_valid), callbacks = [chekcpoint_cb, early_stopping_cb])

# 모델이 향상되지 않으면 훈련이 자동으로 중지되기 때문에 에포크의 숫자를 크게 지정해도 된다
# 이 경우 EarlyStopping 콜백이 훈련이 끝난 후 최상의 가중치를 복원하기 때문에 저장된 모델을 따로 복원할 필요가 없다

# 더 많은 제어를 원한다면 사용자 정의 콜백을 만들 수 있다
# 예를 들어 다음과 같은 사용자 정의 콜백은 훈련하는 동안 검증 손실과 훈련 손실의 비율을 출력한다(즉, 과대적합을 감지한다)

class PrintValTrainRatioCallback(keras.callbacks.Callbacks) :
    def on_epoch_end(self, epoch, logs) :
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


# on_epoch_end 자리에는 on_train_begin(), on_train_end(), on_epoch_begin(), on_epoch_end(), on_batch_begin(), on_batch_end()를 구현할 수 있다
# 필요하다면 콜백은 검증과 예측 단계에서도 사용할 수 있다(예를 들어 디버깅을 위해)
# 평가에 사용하려면 on_test_begin(), on_test_end(), on_test_batch_begin(), on_test_batch_end()를 구현해야 한다(evaluate()에서 호출된다)
# 예측에 상요하려면 on_predict_begin(), on_predict_end(), on_predict_batch_begin(), on_predict_batch_end()를 구현해야 한다

