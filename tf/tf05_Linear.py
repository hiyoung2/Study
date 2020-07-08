import tensorflow as tf

tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [3, 5, 7] # y = 2x + 1

# random_normal()에 대해 찾아보시오
# tnsorflow random_normal() : 0 ~ 1 사이의 정규확률분포 값을 생성해주는 함수
# 원하는 shape 대로 만들어준다
# 랜덤으로 표준분포에 따라 값을 배정하고 싶을 때 사용한다
# 만들고 싶은 형태와 평균, 편차 등을 지정하여 랜덤하게 지정할 수 있다

# ex) x = tf.Variable(tf.random_normal([784, 200], mean = 1, stddev = 0.35))
# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

# normal : normalization , 변수 일반화? -> 정규분포
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# sess = tf.Session()
# print(sess.run(W)) 
# PreconditionError: Attempting to use uninitialized value weight [[{{node _retval_weight_0_0}}]]
# Error 발생
# variable 쓰려면 초기화를 해야 한다!
# sess.run(tf.global_variables_initializer()) # 초기화 작업
# print(sess.run(W))
# 현재 이 소스에서는 하단에 with문을 통해서 모든 변수들을 초기화하는 작업을 한 번에 처리했기 때문에
# 이 단계에서 초기화할 필요가 없으므로 주석 처리

# y = wx + b
hypothesis = x_train * W + b

# 케라스와 어떻게 matching 되는지 생각하기
# maen((hypo - y_train)^2) = mse!
# cost = mse 라는 것을 코딩으로 명시(한땀한땀)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
# 이게 계속 반복되면서 값이 좋아진다(training을 거치면서)
# train.optimizer.학습률.최소화하는방법

# with문 -> 공부 필요
# with : 블록을 지정하는 느낌
# 1. 객체가 생성된 후 with 블럭에 진입하면서 미리 지정된 특정한 직업을 수행한다
# 2. with 블럭을 떠나는 시점에 미리 지정된 특정한 작업을 수행하다

# with문 안에 있는 것들이 모두  Session() 통해 실행됨
# with tf.Session() as sess :
with tf.compat.v1.Session() as sess : # warning 문제 해결 # 위보다 이것을 권장한다는 warning임(compile에는 문제가 없음)
    # sess.run(tf.global_variables_initializer()) # 설정된 전체 변수들이 모두 초기화 된다
    sess.run(tf.compat.v1.global_variables_initializer()) # 이렇게 쓰면 warning 해결
    for step in range(2001) : # 범위 설정 : epochs = 2000
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b]) # keras로 따지면 compile에 해당하는 부분
        # -, : train 값은 따로 출력하지 않음(결과를 보여주지 않는다)
        # cost_val : cost 값 출력
        # W_val : W 값 출력
        # b_val : b 값 출력

        if step % 20 == 0 : # 2000번 돌아가는데 20번째마다 출력
            print(step, cost_val, W_val, b_val) # 횟수, cost(loss라 보면 되고), W, b 

                # 2000 1.21343355e-05 [1.0040361] [-0.00917497]  순서 matching     

'''
x_train = [1, 2, 3]
y_train = [3, 5, 7] # y = 2x + 1
의도한 weight 값 : 2, bias  값 : 1
1940 3.239411e-05 [2.0065947] [0.9850087]
1960 2.9422925e-05 [2.0062847] [0.98571324]
1980 2.6721513e-05 [2.0059893] [0.98638463]
2000 2.4268587e-05 [2.005708] [0.98702455]
횟수     cost        weight        bias
'''