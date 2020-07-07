# tensorflow 1.14 ver 배운다
# 환경 설정 : tf114 가상환경으로 설정

# Tensor? 3차원 이상을 말함

import tensorflow as tf
# tensorflow version check
print(tf.__version__) # 1.14.0

hello = tf.constant("Hello World") # constant : 상수(변하지 않는 값)
print(hello)
# Tensor("Const:0", shape=(), dtype=string)
# 그냥 hello 의 Tensor 자료형이 나온다

# 우리가 아는 형태로, 눈에 보이게 출력하고 싶으면 Session을 통과시켜야 한다
# sess.run 항상 주의!
sess = tf.Session()
print(sess.run(hello))

# (1) from keras.layers import ~
# : keras를 사용, backend로 tensorflow를 사용
# (2) from tensorflow.keras.layers import ~
# : tensorflow 직접 발동, (1)보다 속도가 빠르다

# tensorflow 기본 작동 방식
# input -> tf.Session() -> output
# Session 이라는 어떤 통(?)에 들어가서 연산이 이루어진다, 이 과정을 한 번 거쳐서 output 도출
# Session 을 거치지 않고 바로 간단하게 나올 수 있도록 하는 것이 Keras(backend : Tensorflow)
# TensorFlow 2.xx ver 에서는 Session을 빼 버림
