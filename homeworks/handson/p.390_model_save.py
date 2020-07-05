# 10.2.6 모델 저장과 복원

# 시퀀셜 API와 함수형 API를 사용하면 훈련된 케라스 모델을 저장하는 것은 매우 쉽다

# model = keras.modesl.Sequential([...]) # 또는 keras.Model([...])
# model.compile([...])
# model.fit([...])
# model.save("my_keras_model.h5")

# 케라스는 HDF5 format을 사용하여 (모든 층의 하이퍼파라미터를 포함하여) 모델 구조와 층의 모든 모델 파라미터(즉, 연결 가중치와 편향)을 저장한다
# 또한, (하이퍼파라미터와 현재 상태를 포함하여) 옵티마이저도 저장한다

# 일반적으로 하나의 파이썬 스크립트에서 모델을 훈련하고 저장한 다음 하나 이상의 스크립트(또는 웹 서비스)에서 모델을 로드하고 예측을 만드는 데에 활용한다
# 모델 로드는 다음과 같다

# model = keras.models.load_model("my_keras_model.h5")

# CAUTION! 
# 시퀀셜, 함수형 API에서는 이 방식을 쓸 수 있지만 model subclassing 에서는 사용할 수 없다
# save_weights()와 load_weights() 메서드를 사용하여 모델 파라미터를 저장하고 복원할 수 있다
# 그 외에는 모두 수동으로 저장하고 복원해야 한다
# 서브클래싱 API를 사용한 모델을 저장하는 대안으로 파이썬의 피클(pickle) module을 사용하여 모델 객체를 직렬화 할 수 있다

# 훈련이 몇 시간 동안 지속되는 경우에는 어떻게?
# 대규모 데이터셋에서 훈련할 때 흔히 있는 일인데, 이 경우 컴퓨터에 문제가 생겨 모든 것을 잃지 않으려면
# 훈련 마지막에 모델을 저장하는 것뿐만 아니라 훈련 도중 일정 간격으로 체크포인트(checkpoint)를 저장해야 한다
# 어떻게 fit() 메서드에서 체크포인트를 저장할 수 있을까? 
# 콜백, callback을 사용하면 된다

# 체크포인트 : 텐서플로우에서 모델 파라미터를 저장하는 포맷
# 케라스 모델의 save_weights() 메서드는 기본적으로 체크포인트 포맷을 사용하여 모델 파라미터를 저장한다
# save_format 매개변수를 h5로 지정하거나 파일 경로가 .h5로 끝나는 경우에는 HDF5 포맷으로 저장한다