# tensorflow 1.14 ver 배운다
# 환경 설정 : tf114 가상환경으로 설정

import tensorflow as tf
# tensorflow version check
print(tf.__version__) # 1.14.0

hello = tf.constant("Hello World") # constant : 상수(변하지 않는 값)
print(hello)
# Tensor("Const:0", shape=(), dtype=string)
# 그냥 hello 의 Tensor 자료형이 나온다

# 우리가 아는 형태로, 눈에 보이게 출력하고 싶으면 Session을 통과시켜야 한다
sess = tf.Session()
print(sess.run(hello))