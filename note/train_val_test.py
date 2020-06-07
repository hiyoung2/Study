# 딥러닝에서 신경망 모델을 학습하고 평가하기 위해 dataset이 필요하다
# 이 때 dataset의 성질에 맞게 보통 다음의 3가지로 분류한다

# 1. Train set
# 2. Validation set
# 3. Test set

# <Train set>
# 모델을 학습하기 위한 dataset
# 한 가지 명심해야할 중요한 사실은
# "모델을 학습하는 데에는 오직 유일하게 Train datset만 이용한다"
# 보통 train set을 이용해 각기 다른 모델을 서로 다른 epoch로 학습을 시킨다
# 여기서 각기 다른 모델이란 hidden layer 혹은 hyper parameter에 약간씩 변화를 주는 것을 의미

# <Test set & Validation set>
# validation set은 학습이 이미 완료된 모델을 검증하기 위한 dataset이다
# test set은 학습과 검증이 완료된 모델의 성능을 평가하기 위한 dataset이다
# 보통 Train : Test 데이터를 8 : 2 로 나누는데 여기서 Train 데이터 중 일부를
# validation set으로 이용해 결국 Train : Val : Test 를 일반적으로 6 : 2 : 2 로 이용

# validation set와 test set의 공통점은 이 데이터를 통해 모델을 update, 즉 학습을 시키지 않는다는 것
# 이렇게 validation set와 test set은 둘 다 이미 학습을 완료하 ㄴ모델에 대해 평가하고, 학습은 시키지 않는데
# 둘의 차이는 무엇?

# validation set 은 모델을 update, 즉 학습을 시키지 않지만 학습에 '관여'는 한다
# Test set은 학습에 전혀 관여하지 않고 오직 '최종 성능'을 평가하기 위해 쓰인다

# test set은 모델의 '최종 성능'을 평가하기 위해 쓰이며, 
# training 과정에 관여하지 않는다

# validation set은 여러 모델 중에서 최종 모델을 선정하기 위한 
# 성능 평가에 관여한다고 보면 된다
# 따라서 validation set은 training에 관여하게 된다

# 즉, validation set은 training 과정에 관여를 하며, training이 된 여러 가지 모델 중
# 가장 좋은 모델 하나를 고르기 위한 set이다

# test set은 모든 training 과정이 완료된 후에 최종적으로 모델의 성능을 평가하기 위한 set이다

# 만약, 여러 모델을 성능 평가하여 그 중에서 가장 좋은 모델을 선택하고 싶지 않은 경우엔,
# validation set를 만들지 않아도 된다
# BUT, 이 때에는 문제가 생길 것
# (test accuracy를 예측할 수도 없고, 모델 튜닝을 통해 overfitting을 방지할 수도 없다)